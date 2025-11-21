import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    const MaterialApp(debugShowCheckedModeBanner: false, home: YoloPage()),
  );
}

// --- 1. MODIFICATION : On ajoute un champ pour stocker l'image découpée ---
class Recognition {
  final double x1, y1, x2, y2, score;
  String text;
  File? cropFile; // Variable pour stocker la vignette
  bool? isMisplaced; // null=pas vérifié, false=ok, true=mal placé

  Recognition(
    this.x1,
    this.y1,
    this.x2,
    this.y2,
    this.score, {
    this.text = "",
    this.cropFile,
    this.isMisplaced,
  });
}

class YoloPage extends StatefulWidget {
  const YoloPage({super.key});
  @override
  State<YoloPage> createState() => _YoloPageState();
}

class _YoloPageState extends State<YoloPage> {
  File? _image;
  img.Image? _originalImage;
  Interpreter? _interpreter;
  List<Recognition> _recognitions = [];
  bool _isBusy = false;
  String _status = "En attente d'image";

  final int _inputSize = 640;
  final double _confThreshold = 0.25;
  final double _iouThreshold = 0.45;

  final _textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);

  // --- AJOUT 1 : Contrôleur pour le zoom ---
  final TransformationController _transformController =
      TransformationController();
  double _currentScale = 1.0; // On stocke le niveau de zoom ici

  @override
  void initState() {
    super.initState();
    _loadModel();

    // --- AJOUT 2 : On écoute le changement de zoom ---
    _transformController.addListener(() {
      // getMaxScaleOnAxis() nous donne le niveau de zoom actuel
      final newScale = _transformController.value.getMaxScaleOnAxis();
      if (newScale != _currentScale) {
        setState(() {
          _currentScale = newScale;
        });
      }
    });
  }

  @override
  void dispose() {
    _textRecognizer.close();
    _transformController
        .dispose(); // <-- N'oubliez pas de disposer le contrôleur
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/book_label_obb_s2_float32.tflite',
      );
      setState(() => _status = "Modèle prêt");
    } catch (e) {
      print("Erreur modèle: $e");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      final file = File(pickedFile.path);
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);

      setState(() {
        _image = file;
        _originalImage = decoded;
        _recognitions = [];
        _status = "Analyse en cours...";
      });

      if (decoded != null) {
        final oriented = img.bakeOrientation(decoded);
        _runInference(oriented);
      }
    }
  }

  Future<void> _runInference(img.Image image) async {
    if (_interpreter == null) return;
    setState(() => _isBusy = true);

    // 1. Letterboxing
    double ratioX = _inputSize / image.width;
    double ratioY = _inputSize / image.height;
    double scale = min(ratioX, ratioY);

    int newWidth = (image.width * scale).round();
    int newHeight = (image.height * scale).round();

    img.Image resized = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
    );
    img.Image letterboxedImage = img.Image(
      width: _inputSize,
      height: _inputSize,
    );
    img.fill(letterboxedImage, color: img.ColorRgb8(114, 114, 114));

    int dx = (_inputSize - newWidth) ~/ 2;
    int dy = (_inputSize - newHeight) ~/ 2;
    img.compositeImage(letterboxedImage, resized, dstX: dx, dstY: dy);

    // 2. Input
    var input = List.generate(
      1,
      (i) => List.generate(
        _inputSize,
        (y) => List.generate(_inputSize, (x) {
          var pixel = letterboxedImage.getPixel(x, y);
          return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
        }),
      ),
    );

    // 3. Output
    var outputShape = _interpreter!.getOutputTensor(0).shape;
    var output = List.filled(
      outputShape[0] * outputShape[1] * outputShape[2],
      0.0,
    ).reshape(outputShape);
    _interpreter!.run(input, output);

    // 4. Decoding
    List<Recognition> rawRecognitions = [];
    int numAnchors = outputShape[2];

    for (int i = 0; i < numAnchors; i++) {
      double score = output[0][4][i];
      if (score > _confThreshold) {
        double cx = output[0][0][i] * _inputSize;
        double cy = output[0][1][i] * _inputSize;
        double w = output[0][2][i] * _inputSize;
        double h = output[0][3][i] * _inputSize;
        double angle = output[0][5][i];

        double cosA = cos(angle).abs();
        double sinA = sin(angle).abs();
        double localW = w * cosA + h * sinA;
        double localH = w * sinA + h * cosA;

        double x1_640 = cx - (localW / 2);
        double y1_640 = cy - (localH / 2);
        double x2_640 = cx + (localW / 2);
        double y2_640 = cy + (localH / 2);

        double x1 = (x1_640 - dx) / scale;
        double y1 = (y1_640 - dy) / scale;
        double x2 = (x2_640 - dx) / scale;
        double y2 = (y2_640 - dy) / scale;

        rawRecognitions.add(Recognition(x1, y1, x2, y2, score));
      }
    }

    // 1. On nettoie les doublons
    List<Recognition> nmsRecognitions = _nms(rawRecognitions);

    // 2. --- AJOUT ICI : On trie par ordre de lecture (Lignes puis Colonnes) ---
    List<Recognition> sortedRecognitions = _sortByReadingOrder(nmsRecognitions);

    setState(() {
      // On met à jour avec la liste TRIÉE
      _recognitions = sortedRecognitions;
      _status = "${sortedRecognitions.length} cotes détectées. Lecture...";
    });

    if (sortedRecognitions.isNotEmpty) {
      // L'OCR suivra maintenant l'ordre 1, 2, 3, 4...
      await _performOCR(image, sortedRecognitions);
    } else {
      setState(() {
        _isBusy = false;
        _status = "Aucune cote trouvée";
      });
    }
  }

  Future<void> _performOCR(
    img.Image originalImage,
    List<Recognition> boxes,
  ) async {
    final tempDir = await getTemporaryDirectory();

    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];

      // --- ÉTAPE 1 à 4 : GÉOMÉTRIE, DÉCOUPE, TRAITEMENT (Pas de changement) ---
      // On prend les dimensions brutes de la boîte
      double boxW = box.x2 - box.x1;
      double boxH = box.y2 - box.y1;
      // Calcul des coordonnées sécurisées (ne pas sortir de l'image)
      int x = max(0, (box.x1).toInt());
      int y = max(0, (box.y1).toInt());
      int w = min(originalImage.width - x, (boxW).toInt());
      int h = min(originalImage.height - y, (boxH).toInt());

      if (w <= 0 || h <= 0) {
        box.text = "Erreur dim";
        continue;
      }

      // Extraction de la vignette
      img.Image processed = img.copyCrop(
        originalImage,
        x: x,
        y: y,
        width: w,
        height: h,
      );

      // --- ÉTAPE 2 : L'ÉTIREMENT ANAMORPHIQUE (LE SECRET) ---
      if (processed.height > processed.width * 1.2) {
        int newWidth = (processed.height * 0.8).toInt();

        processed = img.copyResize(
          processed,
          width: newWidth,
          height: processed.height,
          interpolation: img.Interpolation.cubic,
        );
      }

      // --- ÉTAPE 3 : UPSCALING (SUPER-RÉSOLUTION) ---
      if (processed.height < 300) {
        processed = img.copyResize(
          processed,
          height: 300,
          interpolation: img.Interpolation.cubic,
        );
      }

      // --- ÉTAPE 4 : LUMIÈRE & CONTRASTE ---
      processed = img.grayscale(processed);
      processed = img.contrast(processed, contrast: 150);

      // Sauvegarde fichier temporaire
      File cropFile = File('${tempDir.path}/temp_crop_$i.jpg');
      await cropFile.writeAsBytes(img.encodeJpg(processed, quality: 100));

      // On sauvegarde l'image traitée pour l'afficher dans la liste (debug visuel)
      box.cropFile = cropFile;

      // --- LOG 1 : AVANT OCR (pour vérifier l'entrée) ---
      print("--- LOG OCR #${i + 1} START ---");
      print("Path: ${cropFile.path}");
      print("Dimensions (traitées): ${processed.width}x${processed.height}");
      // ----------------------------------------------------

      // --- ÉTAPE 5 : OCR AVEC TIMEOUT ---
      try {
        final inputImage = InputImage.fromFilePath(cropFile.path);

        final recognizedText = await _textRecognizer
            .processImage(inputImage)
            .timeout(const Duration(milliseconds: 4500));

        String rawText = recognizedText.text;

        // --- ÉTAPE 6 : NETTOYAGE REGEX DEWEY (Pas de changement ici, mais inclus pour le contexte) ---
        // A. Nettoyage de base (Sauts de ligne -> Espace)
        String cleanText = rawText.replaceAll("\n", " ").trim().toUpperCase();

        // B. Correction CIBLÉE : Lettre -> Chiffre UNIQUEMENT si suivi d'un chiffre
        // Exemple : "O78" devient "078", mais "DRO" reste "DRO".
        if (cleanText.isNotEmpty) {
          cleanText = cleanText.replaceAllMapped(
            RegExp(
              r'([OQDZSBILG])(?=\d)',
            ), // Capture la lettre seulement si un chiffre la suit immédiatement
            (Match m) {
              String letter = m.group(1)!;
              switch (letter) {
                case 'O':
                case 'Q':
                case 'D':
                  return '0';
                case 'I':
                case 'L':
                  return '1';
                case 'Z':
                  return '2';
                case 'S':
                  return '5';
                case 'G':
                  return '6';
                case 'B':
                  return '8';
                default:
                  return letter;
              }
            },
          );
        }

        // C. Remplacer les séparateurs bizarres par des points
        cleanText = cleanText.replaceAll(RegExp(r'[-_,]'), ".");

        // --- NOUVEAU RÉGLAGE : Supprimer les espaces AVANT un point ---
        // Le regex \s+\. signifie "un ou plusieurs espaces suivis d'un point"
        cleanText = cleanText.replaceAll(RegExp(r'\s+\.'), ".");

        // D. Formatage Dewey : Règle des "3 chiffres + point + suite"
        // On utilise replaceAllMapped pour forcer l'insertion du point après les 3 premiers chiffres
        // dès que l'on trouve une séquence de 3 chiffres suivis par d'autres chiffres.
        cleanText = cleanText.replaceAllMapped(
          RegExp(
            r'(\d{3})\s*(\d+)',
          ), // Cherche: 3 chiffres (G1), espace(s) optionnels, puis 1 ou plusieurs chiffres (G2)
          (Match m) => "${m[1]}.${m[2]}",
        );

        // E. Séparation Chiffres / Lettres (Saut de ligne)
        cleanText = cleanText.replaceAllMapped(
          RegExp(r'([0-9.]+)\s*([A-Z]+)'),
          (Match m) => "${m[1]}\n${m[2]}",
        );

        box.text = cleanText.isEmpty ? "?" : cleanText;
        print("LOG OCR #${i + 1} SUCCESS: Résultat Nettoyé: '${box.text}'");
      } catch (e) {
        // --- LOG 2 : EN CAS D'ÉCHEC ---
        // La raison de l'échec (TimeoutException ou autre erreur MLKit)
        print("LOG OCR #${i + 1} FAILURE: Erreur complète: $e");
        box.text = "⚠️";
      }

      print("--- LOG OCR #${i + 1} END ---");
      setState(() {});
    }

    _checkShelfOrder(); // VÉRIFIER LE TRI
    setState(() {
      _isBusy = false;
      _status = "Terminé : ${boxes.length} résultats";
    });
  }

  List<Recognition> _nms(List<Recognition> boxes) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => b.score.compareTo(a.score));
    List<Recognition> selected = [];
    List<bool> active = List.filled(boxes.length, true);

    for (int i = 0; i < boxes.length; i++) {
      if (active[i]) {
        selected.add(boxes[i]);
        for (int j = i + 1; j < boxes.length; j++) {
          if (active[j]) {
            double iou = _calculateIoU(boxes[i], boxes[j]);
            if (iou > _iouThreshold) active[j] = false;
          }
        }
      }
    }
    return selected;
  }

  double _calculateIoU(Recognition boxA, Recognition boxB) {
    double xA = max(boxA.x1, boxB.x1);
    double yA = max(boxA.y1, boxB.y1);
    double xB = min(boxA.x2, boxB.x2);
    double yB = min(boxA.y2, boxB.y2);
    double interArea = max(0, xB - xA) * max(0, yB - yA);
    double boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
    double boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
    return interArea / (boxAArea + boxBArea - interArea);
  }

  List<Recognition> _sortByReadingOrder(List<Recognition> boxes) {
    if (boxes.isEmpty) return [];

    // 1. D'abord, on trie tout verticalement (du Haut vers le Bas)
    boxes.sort((a, b) => a.y1.compareTo(b.y1));

    List<List<Recognition>> rows = [];
    List<Recognition> currentRow = [];

    // Calcul de la hauteur moyenne d'une étiquette
    double avgHeight =
        boxes.map((e) => e.y2 - e.y1).reduce((a, b) => a + b) / boxes.length;

    // --- REGLAGE ICI ---
    // Puisque les étagères sont très espacées, on peut augmenter la tolérance.
    // On dit : "Tant que l'écart vertical est inférieur à 120% de la hauteur d'une étiquette,
    // considère que c'est la même ligne".
    // Cela permet de gérer les étiquettes collées un peu de travers sans créer de fausse ligne.
    double rowTolerance = avgHeight * 1.2;

    for (var box in boxes) {
      if (currentRow.isEmpty) {
        currentRow.add(box);
      } else {
        // On compare la position Y de la box actuelle avec la PREMIÈRE box de la ligne en cours
        double yDiff = (box.y1 - currentRow.first.y1).abs();

        if (yDiff < rowTolerance) {
          // C'est la MÊME ligne (écart faible)
          currentRow.add(box);
        } else {
          // C'est une NOUVELLE ligne (l'écart est grand, c'est l'étagère du dessous)

          // 1. On trie la ligne qu'on vient de finir de GAUCHE à DROITE
          currentRow.sort((a, b) => a.x1.compareTo(b.x1));
          rows.add(currentRow);

          // 2. On démarre la nouvelle ligne
          currentRow = [box];
        }
      }
    }

    // Ne pas oublier la dernière ligne en cours
    if (currentRow.isNotEmpty) {
      currentRow.sort((a, b) => a.x1.compareTo(b.x1));
      rows.add(currentRow);
    }

    // On remet tout à plat dans une seule liste ordonnée :
    // Ligne 1 (G->D), puis Ligne 2 (G->D), etc.
    return rows.expand((element) => element).toList();
  }

  // --- ALGORITHME DE VÉRIFICATION DU TRI ---

  void _checkShelfOrder() {
    if (_recognitions.isEmpty) return;

    // 1. Préparation : On parse tout le monde
    // items contient les objets Dewey ou null si illisible
    List<_DeweyItem?> items = _recognitions
        .map((r) => _parseDewey(r.text))
        .toList();

    // On reset tout le monde à "Mal placé" par défaut pour être sûr
    for (var r in _recognitions) r.isMisplaced = true;

    // Liste des index qu'on considère comme "Bien rangés" (La Chaîne de Confiance)
    List<int> validIndices = [];

    for (var i = 0; i < items.length; i++) {
      var current = items[i];

      // Si illisible, on ignore (reste gris/neutre selon ta logique d'affichage)
      if (current == null) {
        _recognitions[i].isMisplaced = null;
        continue;
      }

      // CAS 1 : C'est le tout premier livre valide
      if (validIndices.isEmpty) {
        validIndices.add(i);
        _recognitions[i].isMisplaced = false; // Validé
        continue;
      }

      // On récupère le dernier livre validé (Le sommet du mur actuel)
      int lastIndex = validIndices.last;
      var lastValid = items[lastIndex]!;

      // CAS 2 : Comparaison normale
      if (current.compareTo(lastValid) >= 0) {
        // Ça monte ou c'est égal -> Tout va bien, on l'ajoute au mur
        validIndices.add(i);
        _recognitions[i].isMisplaced = false;
      } else {
        // CAS 3 : CONFLIT ! (Ça redescend)
        // current < lastValid.
        // Exemple: On a validé [COL], puis [COU]. Maintenant on a [COL ESS].
        // [COL ESS] est plus petit que [COU].

        bool intruderFound = false;

        // On regarde l'avant-dernier validé (Le Grand-Père)
        if (validIndices.length >= 2) {
          int grandParentIndex = validIndices[validIndices.length - 2];
          var grandParent = items[grandParentIndex]!;

          // Est-ce que le livre actuel irait bien après le Grand-Père ?
          // (Est-ce que COL ESS est plus grand que le COL DRO du début ?)
          if (current.compareTo(grandParent) >= 0) {
            // OUI ! Donc c'est le "Père" (lastIndex) qui était un intrus trop grand.

            // 1. On invalide le précédent (qui était vert, on le met rouge)
            _recognitions[lastIndex].isMisplaced = true;

            // 2. On le retire de la chaîne de confiance
            validIndices.removeLast();

            // 3. On valide le courant à la place
            validIndices.add(i);
            _recognitions[i].isMisplaced = false;

            intruderFound = true;
          }
        } else {
          // Cas spécial : Il n'y a qu'un seul livre validé avant (le tout premier).
          // Si le livre actuel est plus petit que le tout premier,
          // c'est peut-être le tout premier qui est l'intrus !
          // Mais c'est risqué. Pour l'instant, on suppose que le 1er est toujours juste.
          // Si tu veux être strict : on ne fait rien, le livre courant est marqué faux.
        }

        if (!intruderFound) {
          // Si on n'a pas trouvé d'intrus avant, c'est que le livre actuel est vraiment mal rangé.
          _recognitions[i].isMisplaced = true;
        }
      }
    }

    setState(() {});
  }

  // Utilitaire pour parser et comparer
  _DeweyItem? _parseDewey(String text) {
    // Nettoyage : sauts de ligne deviennent espaces, trim
    String clean = text.replaceAll("\n", " ").trim();

    // Regex : Groupe 1 = Chiffre, Groupe 2 = TOUT LE RESTE
    final reg = RegExp(r'^([0-9.]+)\s*(.*)$');
    final match = reg.firstMatch(clean);

    if (match == null) return null;

    try {
      double number = double.parse(match.group(1)!);

      // On récupère la partie texte (ex: "DIR TES" ou "COD CIV")
      String rawSuffix = match.group(2) ?? "";

      // On découpe par les espaces pour avoir les blocs séparés
      // ex: ["DIR", "TES"] ou ["COD", "CIV"] ou ["DIR"]
      List<String> parts = rawSuffix.trim().split(RegExp(r'\s+'));

      String p1 = parts.isNotEmpty ? parts[0] : "";
      String p2 = parts.length > 1 ? parts[1] : ""; // Le 2ème bloc (ex: TES)

      return _DeweyItem(number, p1, p2);
    } catch (e) {
      return null;
    }
  }

  // --- LA MÉTHODE BUILD DOIT ÊTRE ICI (DANS _YoloPageState) ---
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Scanner de Cotes"),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
      ),
      body: Column(
        children: [
          // --- PARTIE 1 : L'IMAGE (50% de l'écran) ---
          Expanded(
            flex: 50,
            child: Container(
              color: Colors.black,
              child: Center(
                child: _image == null
                    ? Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: const [
                          Icon(Icons.image, color: Colors.grey, size: 80),
                          SizedBox(height: 10),
                          Text(
                            "Aucune image sélectionnée",
                            style: TextStyle(color: Colors.grey),
                          ),
                        ],
                      )
                    : LayoutBuilder(
                        builder: (context, constraints) {
                          return InteractiveViewer(
                            // --- AJOUT 3 : Lier le contrôleur ---
                            transformationController: _transformController,
                            panEnabled: true,
                            boundaryMargin: const EdgeInsets.all(0),
                            minScale: 1.0,
                            maxScale: 10.0, // J'ai augmenté un peu le max
                            child: Stack(
                              fit: StackFit.expand,
                              children: [
                                Image.file(_image!, fit: BoxFit.contain),
                                if (_recognitions.isNotEmpty)
                                  CustomPaint(
                                    painter: OcrPainter(
                                      recognitions: _recognitions,
                                      imageSize: Size(
                                        _originalImage!.width.toDouble(),
                                        _originalImage!.height.toDouble(),
                                      ),
                                      widgetSize: Size(
                                        constraints.maxWidth,
                                        constraints.maxHeight,
                                      ),
                                      // --- AJOUT 4 : Passer l'échelle actuelle ---
                                      scale: _currentScale,
                                    ),
                                    child: const SizedBox.expand(),
                                  ),
                              ],
                            ),
                          );
                        },
                      ),
              ),
            ),
          ),

          // --- PARTIE 2 : LA LISTE AVEC IMAGES (50% de l'écran) ---
          Expanded(
            flex: 50,
            child: Column(
              children: [
                if (_isBusy)
                  const LinearProgressIndicator(color: Colors.indigo),

                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(
                    _status,
                    style: const TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.indigo,
                    ),
                  ),
                ),

                // La Liste
                Expanded(
                  child: _recognitions.isEmpty
                      ? const Center(
                          child: Text("Les résultats s'afficheront ici."),
                        )
                      : ListView.separated(
                          padding: const EdgeInsets.symmetric(horizontal: 10),
                          itemCount: _recognitions.length,
                          separatorBuilder: (ctx, i) => const Divider(),
                          itemBuilder: (context, index) {
                            final rec = _recognitions[index];
                            return Card(
                              elevation: 3,
                              margin: const EdgeInsets.symmetric(vertical: 6),
                              child: Padding(
                                padding: const EdgeInsets.all(8.0),
                                child: Row(
                                  crossAxisAlignment: CrossAxisAlignment.start,
                                  children: [
                                    // --- 3. MODIFICATION : Affichage de l'image découpée ---
                                    Container(
                                      width: 60,
                                      height: 80,
                                      decoration: BoxDecoration(
                                        color: Colors.grey[200],
                                        border: Border.all(
                                          color: Colors.indigo.shade100,
                                        ),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: rec.cropFile != null
                                          ? ClipRRect(
                                              borderRadius:
                                                  BorderRadius.circular(8),
                                              child: Image.file(
                                                rec.cropFile!,
                                                fit: BoxFit.contain,
                                              ),
                                            )
                                          : const Center(
                                              child: CircularProgressIndicator(
                                                strokeWidth: 2,
                                              ),
                                            ),
                                    ),

                                    const SizedBox(width: 15),

                                    // Colonne avec les infos
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        children: [
                                          // Badge Numéro
                                          Container(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 8,
                                              vertical: 2,
                                            ),
                                            decoration: BoxDecoration(
                                              color: Colors.indigo,
                                              borderRadius:
                                                  BorderRadius.circular(12),
                                            ),
                                            child: Text(
                                              "Cote #${index + 1}",
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 12,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                          ),
                                          const SizedBox(height: 5),

                                          // Texte OCR
                                          Text(
                                            rec.text.isEmpty
                                                ? "Analyse..."
                                                : rec.text,
                                            style: const TextStyle(
                                              fontWeight: FontWeight.bold,
                                              fontSize: 18,
                                            ),
                                          ),
                                          const SizedBox(height: 5),

                                          // Confiance
                                          Text(
                                            "Score détection: ${(rec.score * 100).toInt()}%",
                                            style: TextStyle(
                                              color: Colors.grey[600],
                                              fontSize: 12,
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),

                                    // Icône de statut de tri
                                    if (rec.isMisplaced == true)
                                      const Tooltip(
                                        message:
                                            "Mal placé ! (Devrait être avant le précédent)",
                                        child: Icon(
                                          Icons.warning_amber_rounded,
                                          color: Colors.red,
                                          size: 30,
                                        ),
                                      )
                                    else if (rec.isMisplaced == false)
                                      const Icon(
                                        Icons.check_circle,
                                        color: Colors.green,
                                        size: 30,
                                      )
                                    else
                                      // Cas où le texte est vide ou non parsable
                                      const Icon(
                                        Icons.help_outline,
                                        color: Colors.grey,
                                      ),
                                  ],
                                ),
                              ),
                            );
                          },
                        ),
                ),
              ],
            ),
          ),

          Container(
            padding: const EdgeInsets.symmetric(vertical: 10),
            color: Colors.grey[200],
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _isBusy
                      ? null
                      : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Caméra"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.indigo,
                    foregroundColor: Colors.white,
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _isBusy
                      ? null
                      : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text("Galerie"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.indigo,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// Petite classe helper pour comparer facilement (à mettre hors de la classe State ou dedans)
class _DeweyItem implements Comparable<_DeweyItem> {
  final double number;
  final String part1; // ex: COD ou DIR
  final String part2; // ex: CIV ou TES (peut être vide)

  _DeweyItem(this.number, this.part1, this.part2);

  @override
  int compareTo(_DeweyItem other) {
    // 1. Comparer les numéros (348 vs 372)
    int numComp = number.compareTo(other.number);
    if (numComp != 0) return numComp;

    // 2. Comparer le premier mot (COD vs DIR)
    int p1Comp = part1.compareTo(other.part1);
    if (p1Comp != 0) return p1Comp;

    // 3. Comparer le second mot (ABC vs TES)
    // "rien" (vide) est considéré comme "plus petit" que "quelque chose"
    return part2.compareTo(other.part2);
  }
}

class OcrPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final Size widgetSize;
  final double scale; // --- Variable reçue du contrôleur ---

  OcrPainter({
    required this.recognitions,
    required this.imageSize,
    required this.widgetSize,
    required this.scale, // --- Requis ---
  });

  @override
  void paint(Canvas canvas, Size size) {
    // Calcul des ratios (inchangé)
    double renderedWidth, renderedHeight;
    double ratioImage = imageSize.width / imageSize.height;
    double ratioScreen = size.width / size.height;

    if (ratioImage > ratioScreen) {
      renderedWidth = size.width;
      renderedHeight = size.width / ratioImage;
    } else {
      renderedHeight = size.height;
      renderedWidth = size.height * ratioImage;
    }

    double scaleX = renderedWidth / imageSize.width;
    double scaleY = renderedHeight / imageSize.height;
    double offsetX = (size.width - renderedWidth) / 2;
    double offsetY = (size.height - renderedHeight) / 2;

    // --- CALCUL DYNAMIQUE DES TAILLES ---
    // Plus le 'scale' est grand (zoom), plus on divise la taille pour qu'elle reste visuellement fine.
    // On définit une épaisseur de base de 3.0 et une police de 14.0
    double strokeWidth = 3.0 / scale;
    double fontSize = 14.0 / scale;

    // On fixe une limite min pour que ça ne disparaisse pas complètement si on dézoome trop
    if (strokeWidth < 1.0) strokeWidth = 1.0;
    if (fontSize < 8.0) fontSize = 8.0;

    for (var i = 0; i < recognitions.length; i++) {
      var rec = recognitions[i];

      // --- LOGIQUE DE COULEUR ---
      Color boxColor;
      if (rec.isMisplaced == true) {
        boxColor = Colors.red; // Mal placé -> Rouge
      } else if (rec.isMisplaced == false) {
        boxColor = Colors.greenAccent; // Bien placé -> Vert
      } else {
        boxColor = Colors.blueAccent; // Pas encore vérifié -> Bleu (optionnel)
      }

      // Définition du style avec la largeur dynamique
      final Paint boxPaint = Paint()
        ..color = boxColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth;

      double left = (rec.x1 * scaleX) + offsetX;
      double top = (rec.y1 * scaleY) + offsetY;
      double right = (rec.x2 * scaleX) + offsetX;
      double bottom = (rec.y2 * scaleY) + offsetY;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), boxPaint);

      // --- DESSIN DU CHIFFRE ---
      TextSpan span = TextSpan(
        style: TextStyle(
          color: boxColor, // Le texte prend la couleur de la boîte
          fontSize: fontSize, // Taille dynamique
          fontWeight: FontWeight.bold,
          // Petit contour noir autour du texte pour lisibilité si fond clair
          shadows: [
            Shadow(offset: Offset(-1 / scale, -1 / scale), color: Colors.black),
            Shadow(offset: Offset(1 / scale, -1 / scale), color: Colors.black),
            Shadow(offset: Offset(1 / scale, 1 / scale), color: Colors.black),
            Shadow(offset: Offset(-1 / scale, 1 / scale), color: Colors.black),
          ],
        ),
        text: "${i + 1}",
      );

      TextPainter tp = TextPainter(
        text: span,
        textDirection: TextDirection.ltr,
      );
      tp.layout();

      // On dessine le numéro au coin en haut à gauche de la boîte
      tp.paint(canvas, Offset(left, top - (fontSize + 2)));
    }
  }

  @override
  bool shouldRepaint(covariant OcrPainter oldDelegate) {
    // On redessine si la liste change OU si le zoom change
    return oldDelegate.recognitions != recognitions ||
        oldDelegate.scale != scale;
  }
}
