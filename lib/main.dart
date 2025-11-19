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
  File? cropFile; // <--- Nouvelle variable pour stocker la vignette

  Recognition(
    this.x1,
    this.y1,
    this.x2,
    this.y2,
    this.score, {
    this.text = "",
    this.cropFile,
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

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  @override
  void dispose() {
    _textRecognizer.close();
    super.dispose();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/book_label_obb_s2_float32.tflite',
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

      // --- 1. DÉCOUPE SÉCURISÉE ---
      int x = max(0, box.x1.toInt());
      int y = max(0, box.y1.toInt());
      int w = min(originalImage.width - x, (box.x2 - box.x1).toInt());
      int h = min(originalImage.height - y, (box.y2 - box.y1).toInt());

      if (w <= 0 || h <= 0) continue;

      // On découpe
      img.Image cropped = img.copyCrop(
        originalImage,
        x: x,
        y: y,
        width: w,
        height: h,
      );

      // --- 2. AMÉLIORATIONS D'IMAGE (LA CLÉ DU SUCCÈS) ---

      // A. Rotation intelligente : Si l'image est verticale (hauteur > 1.2x largeur)
      // On suppose que le texte est écrit de bas en haut (standard bibliothèque)
      if (cropped.height > cropped.width * 1.2) {
        cropped = img.copyRotate(
          cropped,
          angle: -90,
        ); // Tourne de 90° vers la droite
      }

      // B. Noir et Blanc + Contraste
      // Cela enlève la couleur des livres pour ne garder que le texte
      cropped = img.grayscale(cropped);
      cropped = img.contrast(
        cropped,
        contrast: 150,
      ); // Augmente le contraste (100 = normal)

      // C. Padding (Ajout de bords blancs)
      // On crée une image un peu plus grande et on colle le crop au milieu
      int padding = 10;
      img.Image padded = img.Image(
        width: cropped.width + (padding * 2),
        height: cropped.height + (padding * 2),
      );
      img.fill(padded, color: img.ColorRgb8(255, 255, 255)); // Fond blanc
      img.compositeImage(padded, cropped, dstX: padding, dstY: padding);

      // C'est cette image finale optimisée qu'on sauvegarde
      File cropFile = File('${tempDir.path}/temp_crop_$i.jpg');
      await cropFile.writeAsBytes(
        img.encodeJpg(padded),
      ); // On sauvegarde 'padded'

      // Sauvegarde pour affichage UI
      box.cropFile = cropFile;

      // --- 3. LECTURE OCR ---
      final inputImage = InputImage.fromFilePath(cropFile.path);
      final recognizedText = await _textRecognizer.processImage(inputImage);

      // --- 4. NETTOYAGE DU TEXTE (REGEX) ---
      // On remplace les sauts de ligne par des espaces
      String rawText = recognizedText.text.replaceAll("\n", " ").trim();

      // OPTIONNEL : Filtrage strict (Garder seulement Majuscules et Chiffres)
      // Si vos cotes ressemblent à "823.91 ROW", ceci enlèvera le bruit
      // rawText = rawText.replaceAll(RegExp(r'[^A-Z0-9. ]'), '');

      box.text = rawText;
      setState(() {});
    }

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
                          return Stack(
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
                                  ),
                                  child: const SizedBox.expand(),
                                ),
                            ],
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
                                                fit: BoxFit.cover,
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

                                    // Icône de validation
                                    if (rec.text.isNotEmpty)
                                      const Icon(
                                        Icons.check_circle,
                                        color: Colors.green,
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

class OcrPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final Size widgetSize;

  OcrPainter({
    required this.recognitions,
    required this.imageSize,
    required this.widgetSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boxPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

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

    for (var i = 0; i < recognitions.length; i++) {
      var rec = recognitions[i];
      double left = (rec.x1 * scaleX) + offsetX;
      double top = (rec.y1 * scaleY) + offsetY;
      double right = (rec.x2 * scaleX) + offsetX;
      double bottom = (rec.y2 * scaleY) + offsetY;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), boxPaint);

      // Numéro sur l'image pour faire le lien avec la liste
      TextSpan span = TextSpan(
        style: const TextStyle(
          color: Colors.white,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),
        text: "${i + 1}",
      );
      TextPainter tp = TextPainter(
        text: span,
        textDirection: TextDirection.ltr,
      );
      tp.layout();

      Paint circlePaint = Paint()..color = Colors.indigo;
      canvas.drawCircle(Offset(left, top), 12, circlePaint);
      tp.paint(canvas, Offset(left - tp.width / 2, top - tp.height / 2));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
