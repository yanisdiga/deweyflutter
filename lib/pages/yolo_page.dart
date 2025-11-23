import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';

// Imports de nos nouveaux fichiers
import '../models/recognition.dart';
import '../models/dewey_item.dart';
import '../widgets/ocr_painter.dart';
import '../utils/box_utils.dart';

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
  bool _showArrows = true;

  // Paramètres YOLO
  final int _inputSize = 640;
  final double _confThreshold = 0.25;
  final double _iouThreshold = 0.45;

  // --- MOTEUR OCR : GOOGLE ML KIT ---
  final TextRecognizer _textRecognizer = TextRecognizer(
    script: TextRecognitionScript.latin,
  );

  final TransformationController _transformController =
      TransformationController();
  double _currentScale = 1.0;

  @override
  void initState() {
    super.initState();
    _loadYoloModel();
    _transformController.addListener(() {
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
    _interpreter?.close();
    _transformController.dispose();
    super.dispose();
  }

  Future<void> _loadYoloModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/book_label_obb_s2_float32.tflite',
      );
      setState(() => _status = "Modèle YOLO prêt");
    } catch (e) {
      print("Erreur modèle YOLO: $e");
      setState(() => _status = "Erreur chargement modèle");
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: source);

    if (pickedFile != null) {
      // --- CORRECTION : ON VIDE TOUT IMMÉDIATEMENT ---
      setState(() {
        _isBusy = true;
        _recognitions = []; // On vide la liste des cotes
        _image = File(
          pickedFile.path,
        ); // On affiche la nouvelle image (chargement)
        _status = "Décodage de l'image...";
      });

      // Le décodage peut être long, l'interface est maintenant vide en attendant
      final file = File(pickedFile.path);
      final bytes = await file.readAsBytes();
      final decoded = img.decodeImage(bytes);

      setState(() {
        _originalImage = decoded;
        _status = "Analyse YOLO en cours...";
      });

      if (decoded != null) {
        final oriented = img.bakeOrientation(decoded);
        await _runYoloInference(oriented);
      }
    }
  }

  Future<void> _runYoloInference(img.Image image) async {
    if (_interpreter == null) return;
    setState(() => _isBusy = true);

    // A. Letterboxing
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

    // B. Input
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

    // C. Output
    var outputShape = _interpreter!.getOutputTensor(0).shape;
    // outputShape est généralement [1, 6, 8400] pour YOLO OBB (Batch, Channels, Anchors)

    // CORRECTION : On génère la liste imbriquée directement au lieu d'utiliser reshape
    // Cela crée une structure [1][Channels][Anchors] remplie de 0.0
    var output = List.generate(outputShape[0], (batch) {
      return List.generate(outputShape[1], (channel) {
        return List.filled(outputShape[2], 0.0);
      });
    });

    _interpreter!.run(input, output);

    // D. Decoding
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

    // E. Nettoyage via BoxUtils
    List<Recognition> nmsRecognitions = BoxUtils.nms(
      rawRecognitions,
      _iouThreshold,
    );
    List<Recognition> sortedRecognitions = BoxUtils.sortByReadingOrder(
      nmsRecognitions,
    );

    setState(() {
      _recognitions = sortedRecognitions;
      _status =
          "${sortedRecognitions.length} zones détectées. Lecture ML Kit...";
    });

    if (sortedRecognitions.isNotEmpty) {
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
      double boxW = box.x2 - box.x1;
      double boxH = box.y2 - box.y1;
      int x = max(0, (box.x1).toInt());
      int y = max(0, (box.y1).toInt());
      int w = min(originalImage.width - x, (boxW).toInt());
      int h = min(originalImage.height - y, (boxH).toInt());

      if (w <= 0 || h <= 0) {
        box.text = "Erreur dim";
        continue;
      }

      img.Image processed = img.copyCrop(
        originalImage,
        x: x,
        y: y,
        width: w,
        height: h,
      );

      if (processed.height > processed.width * 1.2) {
        int newWidth = (processed.height * 0.8).toInt();
        processed = img.copyResize(
          processed,
          width: newWidth,
          height: processed.height,
          interpolation: img.Interpolation.cubic,
        );
      }
      if (processed.height < 300) {
        processed = img.copyResize(
          processed,
          height: 300,
          interpolation: img.Interpolation.cubic,
        );
      }
      processed = img.grayscale(processed);
      processed = img.contrast(processed, contrast: 150);

      String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      File cropFile = File('${tempDir.path}/crop_${timestamp}_$i.jpg');

      await cropFile.writeAsBytes(img.encodeJpg(processed, quality: 100));
      box.cropFile = cropFile;

      try {
        final inputImage = InputImage.fromFilePath(cropFile.path);
        final recognizedText = await _textRecognizer
            .processImage(inputImage)
            .timeout(const Duration(milliseconds: 3000));
        String rawText = recognizedText.text;

        // --- DEBUT NETTOYAGE OPTIMISÉ ---

        // 1. Normalisation de base (Sauts de ligne -> Espaces, Majuscules, Trim)
        String cleanText = rawText.replaceAll("\n", " ").trim().toUpperCase();

        // 2. Correction optique des confusions Chiffres/Lettres
        // On ne corrige QUE si la lettre est suivie d'un chiffre (pour ne pas casser le nom de l'auteur)
        if (cleanText.isNotEmpty) {
          cleanText = cleanText.replaceAllMapped(
            RegExp(r'([OQDZSBILG])(?=\d)'),
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

        // 3. Uniformisation des séparateurs (tirets, underscores -> points)
        cleanText = cleanText.replaceAll(RegExp(r'[-_,]'), ".");

        // 4. Nettoyage spécifique : Suppression d'une lettre isolée au début si suivie de chiffres
        // Ex: "Y | 320 .94" -> "320 .94"
        // Ex: "A 320" -> "320"
        cleanText = cleanText.replaceAll(
          RegExp(r'^[A-Z]\s*[|.]?\s*(?=\d{3})'),
          "",
        );

        // 5. Nettoyage spécifique : Suppression d'une lettre isolée entre des chiffres et un point
        // Ex: "612 F .7" -> "612.7"
        cleanText = cleanText.replaceAllMapped(
          RegExp(r'(\d+)\s+[A-Z]\s+(\.\d+)'),
          (Match m) => "${m[1]}${m[2]}",
        );

        // --- LOGIQUE DE RECONSTRUCTION ---

        // CAS A : Cote FICTION (Commence par une Lettre : R, BD, J, SF...)
        if (RegExp(r'^[A-Z]').hasMatch(cleanText)) {
          // On sépare le préfixe (ex: "BD") du reste (ex: "TINTIN")
          // Regex : Un bloc de lettres/chiffres (le genre), espace, le reste
          cleanText = cleanText.replaceAllMapped(
            RegExp(r'^([A-Z0-9]+)\s+(.*)$'),
            (Match m) => "${m[1]}\n${m[2]}", // "BD\nTINTIN"
          );
        }
        // CAS B : Cote DEWEY (Commence par 3 Chiffres : 632...)
        else if (RegExp(r'^\d{3}').hasMatch(cleanText)) {
          String numberZone = "";
          String authorZone = "";

          // ÉTAPE 1 : SÉPARATION INTELLIGENTE
          // On cherche la frontière : Des chiffres/points d'un côté, des lettres de l'autre.
          // On utilise un "Lookahead" pour trouver le premier bloc de lettres significatif (au moins 1 lettre)

          final splitMatch = RegExp(
            r'^([\d\s\.]+)\s+([A-Z].*)$',
          ).firstMatch(cleanText);

          if (splitMatch != null) {
            numberZone = splitMatch.group(1) ?? ""; // Ex: "632 .012"
            authorZone = splitMatch.group(2) ?? ""; // Ex: "TES SEC S"
          } else {
            // Si pas de séparation claire, tout est considéré comme nombre (cas rare/erreur)
            numberZone = cleanText;
          }

          // ÉTAPE 2 : NETTOYAGE ZONE NUMÉRIQUE (Dewey Strict)
          // On supprime TOUTES les lettres qui auraient pu se glisser dans les chiffres
          numberZone = numberZone.replaceAll(RegExp(r'[A-Z]'), "");
          // On supprime TOUS les espaces
          String cleanDigits = numberZone.replaceAll(" ", ""); // "632.012"
          // On enlève les points pour les remettre au bon endroit
          String digitsOnly = cleanDigits.replaceAll(".", ""); // "632012"

          if (digitsOnly.length >= 3) {
            String classNum = digitsOnly.substring(0, 3);
            // Si plus de 3 chiffres, on FORCE le point après le 3ème
            if (digitsOnly.length > 3) {
              String sub = digitsOnly.substring(3);
              cleanDigits = "$classNum.$sub"; // "632.012"
            } else {
              cleanDigits = classNum; // "632"
            }
          }

          // ÉTAPE 3 : NETTOYAGE ZONE AUTEUR (Suppression du bruit)
          authorZone = authorZone.trim();

          // A. Suppression des lettres isolées à la fin (Bruit OCR)
          // Ex: "TES SEC S" -> Le "S" est supprimé.
          // Mais "T.1" ou "V 2" ne sont PAS supprimés car suivis de chiffres.
          // Regex : Un espace + UNE lettre majuscule + Fin de chaine
          authorZone = authorZone.replaceAll(RegExp(r'\s+[A-Z]$'), "");

          // B. Normalisation des Tomes/Volumes (ex: "T 1" -> "T.1")
          // Si on trouve T, V, VOL suivi d'un espace et d'un chiffre
          authorZone = authorZone.replaceAllMapped(
            RegExp(r'\s+(T|V|VOL)\s*(\.?)\s*(\d+)$'),
            (Match m) => " ${m[1]}.${m[3]}", // " T.1"
          );

          // ÉTAPE 4 : ASSEMBLAGE FINAL
          cleanText = "$cleanDigits\n$authorZone";
        }

        box.text = cleanText.isEmpty ? "?" : cleanText;
        // --- FIN NETTOYAGE ---

        box.text = cleanText.isEmpty ? "?" : cleanText;
      } catch (e) {
        box.text = "⚠️";
      }
      setState(() {});
    }

    _checkShelfOrder();
    setState(() {
      _isBusy = false;
      _status = "Terminé : ${boxes.length} résultats";
    });
  }

  // --- ALGORITHME LIS (Longest Increasing Subsequence) ---
  void _checkShelfOrder() {
    if (_recognitions.isEmpty) return;

    // 1. Parsing
    List<DeweyItem> allItems = _recognitions.map((r) {
      // Validation supplémentaire : Lettre suivie d'un chiffre (ex: "A1") est invalide
      // Mais "T.1" ou "vol.1" sont valides (car séparateur)
      if (RegExp(r'[a-zA-Z][0-9]').hasMatch(r.text)) {
        return DeweyItem(
          isNumeric: false,
          prefix: r.text,
          number: 0,
          cutter: "",
          isValid: false,
        );
      }
      return _parseDewey(r.text);
    }).toList();

    // 2. Candidats Valides (Liste verte potentielle)
    List<MapEntry<int, DeweyItem>> candidates = [];
    for (int i = 0; i < allItems.length; i++) {
      if (allItems[i].isValid) candidates.add(MapEntry(i, allItems[i]));
    }

    // 3. LIS (Algorithme de tri - Plus longue sous-suite croissante)
    Set<int> validOriginalIndices = {};
    List<MapEntry<int, DeweyItem>> validSequence =
        []; // La liste verte ordonnée

    if (candidates.isNotEmpty) {
      int n = candidates.length;
      List<int> L = List.filled(n, 1);
      List<int> P = List.filled(n, -1);

      for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) {
          if (candidates[i].value.compareTo(candidates[j].value) >= 0) {
            if (L[j] + 1 > L[i]) {
              L[i] = L[j] + 1;
              P[i] = j;
            }
          }
        }
      }

      int maxLength = 0;
      int endIndex = -1;
      for (int i = 0; i < n; i++) {
        if (L[i] > maxLength) {
          maxLength = L[i];
          endIndex = i;
        }
      }

      // Reconstruction de la chaîne valide
      while (endIndex != -1) {
        validOriginalIndices.add(candidates[endIndex].key);
        validSequence.add(candidates[endIndex]);
        endIndex = P[endIndex];
      }
      // On remet dans l'ordre croissant (car reconstruit à l'envers)
      validSequence = validSequence.reversed.toList();
    }

    // 4. Calcul des suggestions & Construction du Log
    StringBuffer debugLog = StringBuffer();
    debugLog.writeln("\n🔍 --- RÉSUMÉ DU TRI & SUGGESTIONS ---");

    for (int i = 0; i < _recognitions.length; i++) {
      var item = allItems[i];

      // Reset des états
      _recognitions[i].placementSuggestion = null;
      String debugText = _recognitions[i].text.replaceAll('\n', ' | ');
      String statusIcon;

      // A. Cas Invalide
      if (!item.isValid) {
        _recognitions[i].isMisplaced = null;
        _recognitions[i].isInvalid = true;
        statusIcon = "🟠 INVALID";
      }
      // B. Cas Bien Placé (Vert)
      else if (validOriginalIndices.contains(i)) {
        _recognitions[i].isMisplaced = false;
        _recognitions[i].isInvalid = false;
        statusIcon = "🟢 OK";
      }
      // C. Cas Intrus (Rouge) -> On calcule la suggestion
      else {
        _recognitions[i].isMisplaced = true;
        _recognitions[i].isInvalid = false;
        statusIcon = "🔴 INTRUS";

        // LOGIQUE DE SUGGESTION
        // On cherche où cet intrus s'insère dans la "validSequence" (les verts)
        int? targetDisplayNum;
        String directionArrow = "";
        String positionText = "";

        if (validSequence.isEmpty) {
          _recognitions[i].placementSuggestion = "Aucune référence valide";
        } else {
          // On cherche le premier livre vert qui est PLUS GRAND que l'intrus
          int foundIndex = -1;

          for (var validEntry in validSequence) {
            if (item.compareTo(validEntry.value) < 0) {
              // L'intrus est plus petit que ce livre vert -> Il va AVANT lui
              foundIndex = validEntry.key;
              targetDisplayNum = foundIndex + 1;

              // Si l'index cible est plus petit que l'actuel, on remonte (⬆️), sinon on descend (⬇️)
              // (Dans une liste : avant = au-dessus)
              directionArrow = (foundIndex < i) ? "⬆️" : "⬇️";
              positionText = "Avant le #$targetDisplayNum";
              break;
            }
          }

          // Si on n'a rien trouvé, c'est qu'il est plus grand que TOUS les verts
          // Il va APRES le dernier vert
          if (foundIndex == -1) {
            var lastValid = validSequence.last;
            targetDisplayNum = lastValid.key + 1;

            directionArrow = (lastValid.key > i) ? "⬇️" : "⬆️";
            positionText = "Après le #$targetDisplayNum";
          }

          _recognitions[i].placementSuggestion =
              "$directionArrow $positionText";
          statusIcon += " -> ${_recognitions[i].placementSuggestion}";
        }
      }

      debugLog.writeln("   #${i + 1}: [$debugText] -> $statusIcon");
    }

    print(debugLog.toString());
    setState(() {});
  }

  // --- PARSER DEWEY (Strict & Robuste) ---
  DeweyItem _parseDewey(String text) {
    // Si erreur technique ou vide -> On renvoie un Item "Invalide" au lieu de null
    if (text == "⚠️" || text.trim().isEmpty) {
      return DeweyItem(
        isNumeric: false,
        prefix: "?",
        number: 0,
        cutter: "",
        isValid: false, // Sera traité comme invalide (Orange ou Rouge)
      );
    }

    // 2. Découpage
    List<String> lines = text.split('\n');
    // ... Le reste de votre code reste identique ...
    String firstLine = lines.isNotEmpty ? lines[0].trim() : "";
    String secondLine = lines.length > 1 ? lines[1].trim() : "";

    if (secondLine.length < 2) {
      return DeweyItem(
        isNumeric: false,
        prefix: firstLine.isEmpty ? "?" : firstLine,
        number: 0,
        cutter: "",
        isValid: false,
      );
    }

    bool isFiction = RegExp(r'^[A-Z]').hasMatch(firstLine);

    if (isFiction) {
      return DeweyItem(
        prefix: firstLine,
        number: 0.0,
        cutter: secondLine,
        isNumeric: false,
        isValid: true,
      );
    } else {
      try {
        double val = double.parse(firstLine);
        bool valid = (val >= 0 && val < 1000);

        // Vérification : au moins 3 chiffres
        String integerPart = firstLine.split('.')[0];
        if (integerPart.length < 3) valid = false;

        return DeweyItem(
          prefix: "",
          number: val,
          cutter: secondLine,
          isNumeric: true,
          isValid: valid,
        );
      } catch (e) {
        return DeweyItem(
          isNumeric: true,
          prefix: "",
          number: 0,
          cutter: secondLine,
          isValid: false,
        );
      }
    }
  }

  void _showSettingsModal() {
    showModalBottomSheet(
      context: context,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        // StatefulBuilder permet de mettre à jour le switch visuellement dans la modale
        return StatefulBuilder(
          builder: (BuildContext context, StateSetter setModalState) {
            return Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Paramètres d'affichage",
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 20),
                  SwitchListTile(
                    title: const Text("Afficher les flèches de tri"),
                    subtitle: const Text(
                      "Dessine les liens vers la position correcte",
                    ),
                    secondary: const Icon(
                      Icons.compare_arrows,
                      color: Colors.indigo,
                    ),
                    value: _showArrows,
                    activeColor: Colors.indigo,
                    onChanged: (bool value) {
                      // 1. Mise à jour visuelle du switch dans la modale
                      setModalState(() {
                        _showArrows = value;
                      });
                      // 2. Mise à jour de l'écran principal (YoloPage)
                      this.setState(() {});
                    },
                  ),
                  // Vous pourrez ajouter d'autres options ici plus tard...
                  const SizedBox(height: 20),
                ],
              ),
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Scanner Dewey"),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
        actions: [
          // AJOUT DU BOUTON PARAMÈTRE ICI
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: "Paramètres",
            onPressed: _showSettingsModal,
          ),
        ],
      ),
      body: Column(
        children: [
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
                            transformationController: _transformController,
                            panEnabled: true,
                            boundaryMargin: const EdgeInsets.all(0),
                            minScale: 1.0,
                            maxScale: 10.0,
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
                                      scale: _currentScale,
                                      showArrows: _showArrows,
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
                                                // Ajout de la clé pour forcer le refresh
                                                key: ValueKey(
                                                  rec.cropFile!.path,
                                                ),
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
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment:
                                            CrossAxisAlignment.start,
                                        mainAxisAlignment:
                                            MainAxisAlignment.center,
                                        children: [
                                          // 1. Badge du Numéro
                                          Container(
                                            padding: const EdgeInsets.symmetric(
                                              horizontal: 8,
                                              vertical: 2,
                                            ),
                                            decoration: BoxDecoration(
                                              color: Colors.indigo,
                                              borderRadius:
                                                  BorderRadius.circular(4),
                                            ),
                                            child: Text(
                                              "Livre #${index + 1}",
                                              style: const TextStyle(
                                                color: Colors.white,
                                                fontSize: 11,
                                                fontWeight: FontWeight.bold,
                                              ),
                                            ),
                                          ),
                                          const SizedBox(height: 6),

                                          // 2. Le Texte lu (Cote)
                                          Text(
                                            rec.text.isEmpty ? "..." : rec.text,
                                            style: const TextStyle(
                                              fontWeight: FontWeight.w900,
                                              fontSize:
                                                  16, // Un peu plus petit pour gérer les longs textes
                                              color: Colors.black87,
                                            ),
                                            maxLines: 2,
                                            overflow: TextOverflow.ellipsis,
                                          ),

                                          // 3. ZONE D'ACTION (Si mal placé)
                                          if (rec.placementSuggestion != null)
                                            Container(
                                              margin: const EdgeInsets.only(
                                                top: 8,
                                              ),
                                              padding: const EdgeInsets.all(8),
                                              decoration: BoxDecoration(
                                                color: Colors
                                                    .red
                                                    .shade50, // Fond rouge très clair
                                                border: Border.all(
                                                  color: Colors.red.shade300,
                                                ),
                                                borderRadius:
                                                    BorderRadius.circular(6),
                                              ),
                                              child: Row(
                                                children: [
                                                  // L'icône change selon que c'est INVALID ou INTRUS
                                                  Icon(
                                                    Icons
                                                        .move_up_rounded, // Icône de déplacement
                                                    color: Colors.red.shade700,
                                                    size: 20,
                                                  ),
                                                  const SizedBox(width: 8),
                                                  Expanded(
                                                    child: Text(
                                                      rec.placementSuggestion!, // ex: "⬇️ Avant le #10"
                                                      style: TextStyle(
                                                        color:
                                                            Colors.red.shade900,
                                                        fontWeight:
                                                            FontWeight.bold,
                                                        fontSize: 13,
                                                      ),
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            )
                                          // 4. Info confiance (Si pas de suggestion d'erreur)
                                          else
                                            Padding(
                                              padding: const EdgeInsets.only(
                                                top: 6,
                                              ),
                                              child: Row(
                                                children: [
                                                  Icon(
                                                    Icons.analytics_outlined,
                                                    size: 14,
                                                    color: Colors.grey[600],
                                                  ),
                                                  const SizedBox(width: 4),
                                                  Text(
                                                    "Confiance: ${(rec.score * 100).toInt()}%",
                                                    style: TextStyle(
                                                      color: Colors.grey[600],
                                                      fontSize: 12,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                            ),
                                        ],
                                      ),
                                    ),
                                    // --- ICÔNES DE STATUT ---
                                    if (rec.isInvalid)
                                      const Tooltip(
                                        message:
                                            "Format suspect (Vérification humaine)",
                                        child: Icon(
                                          Icons
                                              .priority_high_rounded, // Point d'exclamation
                                          color: Colors.orange,
                                          size: 30,
                                        ),
                                      )
                                    else if (rec.isMisplaced == true)
                                      const Tooltip(
                                        message: "Mal rangé !",
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
