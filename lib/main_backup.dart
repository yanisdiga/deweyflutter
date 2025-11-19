import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

void main() {
  // On s'assure que les bindings sont prêts
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MaterialApp(home: YoloPage()));
}

class Recognition {
  final double x1, y1, x2, y2, score;
  Recognition(this.x1, this.y1, this.x2, this.y2, this.score);

  @override
  String toString() {
    return 'Box[${x1.toStringAsFixed(1)}, ${y1.toStringAsFixed(1)} -> ${x2.toStringAsFixed(1)}, ${y2.toStringAsFixed(1)}] Score:${score.toStringAsFixed(2)}';
  }
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

  final int _inputSize = 640;
  final double _confThreshold = 0.5;
  final double _iouThreshold = 0.45;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/book_label_obb_s2_float32.tflite',
      );
      print("✅ LOG: Modèle chargé.");
    } catch (e) {
      print("❌ LOG: Erreur chargement modèle: $e");
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
      });

      if (decoded != null) {
        print(
          "✅ LOG: Image chargée. Taille réelle: ${decoded.width} x ${decoded.height}",
        );
        // Correction orientation EXIF importante
        final oriented = img.bakeOrientation(decoded);
        _runInference(oriented);
      }
    }
  }

  Future<void> _runInference(img.Image image) async {
    if (_interpreter == null) return;
    setState(() => _isBusy = true);

    // --- ÉTAPE 1 : LETTERBOXING (Respecter le ratio) ---
    // On calcule comment redimensionner sans écraser
    double ratioX = _inputSize / image.width;
    double ratioY = _inputSize / image.height;
    double scale = min(
      ratioX,
      ratioY,
    ); // On prend le ratio le plus petit pour que tout rentre

    int newWidth = (image.width * scale).round();
    int newHeight = (image.height * scale).round();

    // 1. Redimensionner l'image en gardant les proportions
    img.Image resized = img.copyResize(
      image,
      width: newWidth,
      height: newHeight,
    );

    // 2. Créer un fond carré (Gris 114/114/114 comme YOLO, ou Noir 0/0/0)
    img.Image letterboxedImage = img.Image(
      width: _inputSize,
      height: _inputSize,
    );
    img.fill(letterboxedImage, color: img.ColorRgb8(114, 114, 114));

    // 3. Coller l'image redimensionnée au centre (ou en haut à gauche)
    // YOLOv8 par défaut centre souvent, mais le "paste" simple marche aussi.
    // Pour simplifier les maths, on colle en (0,0) ou centré. Centrons-le :
    int dx = (_inputSize - newWidth) ~/ 2;
    int dy = (_inputSize - newHeight) ~/ 2;

    img.compositeImage(letterboxedImage, resized, dstX: dx, dstY: dy);

    // --- ÉTAPE 2 : PRÉPARATION INPUT ---
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

    // --- ÉTAPE 3 : INFÉRENCE ---
    var outputShape = _interpreter!.getOutputTensor(0).shape;
    var output = List.filled(
      outputShape[0] * outputShape[1] * outputShape[2],
      0.0,
    ).reshape(outputShape);
    _interpreter!.run(input, output);

    // --- ÉTAPE 4 : DÉCODAGE (Avec correction du Letterboxing) ---
    List<Recognition> rawRecognitions = [];
    int numAnchors = outputShape[2];

    for (int i = 0; i < numAnchors; i++) {
      double score = output[0][4][i];
      if (score > _confThreshold) {
        // Coordonnées dans le carré 640x640
        double cx = output[0][0][i] * _inputSize;
        double cy = output[0][1][i] * _inputSize;
        double w = output[0][2][i] * _inputSize;
        double h = output[0][3][i] * _inputSize;
        double angle = output[0][5][i];

        // OBB -> XYXY (Boîte droite locale)
        double cosA = cos(angle).abs();
        double sinA = sin(angle).abs();
        double localW = w * cosA + h * sinA;
        double localH = w * sinA + h * cosA;

        // XYXY local dans le carré 640
        double x1_640 = cx - (localW / 2);
        double y1_640 = cy - (localH / 2);
        double x2_640 = cx + (localW / 2);
        double y2_640 = cy + (localH / 2);

        // --- CORRECTION : Remapper vers l'image d'origine ---
        // On retire les bandes noires (dx, dy) et on divise par le scale
        double x1 = (x1_640 - dx) / scale;
        double y1 = (y1_640 - dy) / scale;
        double x2 = (x2_640 - dx) / scale;
        double y2 = (y2_640 - dy) / scale;

        rawRecognitions.add(Recognition(x1, y1, x2, y2, score));
      }
    }

    List<Recognition> finalRecognitions = _nms(rawRecognitions);
    setState(() {
      _recognitions = finalRecognitions;
      _isBusy = false;
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("DEBUG MODE")),
      body: Column(
        children: [
          Expanded(
            child: Container(
              color: Colors.grey[200], // Fond gris pour voir les limites
              child: Center(
                child: _image == null
                    ? const Text("Aucune image")
                    : LayoutBuilder(
                        builder: (context, constraints) {
                          print(
                            "📐 LOG: LayoutBuilder Contraintes: W=${constraints.maxWidth}, H=${constraints.maxHeight}",
                          );

                          return Stack(
                            fit: StackFit.expand, // Force à remplir l'espace
                            children: [
                              Image.file(
                                _image!,
                                fit: BoxFit.contain,
                                // On s'assure que l'image ne cache pas le reste
                              ),

                              if (_recognitions.isNotEmpty)
                                CustomPaint(
                                  painter: DebugBoxPainter(
                                    recognitions: _recognitions,
                                    inputSize: _inputSize.toDouble(),
                                    imageSize: Size(
                                      _originalImage!.width.toDouble(),
                                      _originalImage!.height.toDouble(),
                                    ),
                                    widgetSize: Size(
                                      constraints.maxWidth,
                                      constraints.maxHeight,
                                    ),
                                  ),
                                  child:
                                      const SizedBox.expand(), // Prend toute la place
                                ),
                            ],
                          );
                        },
                      ),
              ),
            ),
          ),
          Text("Détections: ${_recognitions.length}"),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: () => _pickImage(ImageSource.camera),
                child: const Icon(Icons.camera_alt),
              ),
              ElevatedButton(
                onPressed: () => _pickImage(ImageSource.gallery),
                child: const Icon(Icons.photo),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

// --- PAINTER DE DEBUG ---
class DebugBoxPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final double inputSize; 
  final Size imageSize;   // Taille réelle (ex: 4284 x 5712)
  final Size widgetSize;  // Taille écran (ex: 411 x 738)

  DebugBoxPainter({
    required this.recognitions, 
    required this.inputSize, 
    required this.imageSize, 
    required this.widgetSize
  });

  @override
  void paint(Canvas canvas, Size size) {
    final Paint boxPaint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    // 1. Calcul de la zone d'affichage de l'image (Comme BoxFit.contain)
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

    // 2. --- CORRECTION CRUCIALE ICI ---
    // Avant: on divisait par inputSize (640).
    // Maintenant: on divise par imageSize (4284) car les boîtes sont en coords réelles.
    double scaleX = renderedWidth / imageSize.width;
    double scaleY = renderedHeight / imageSize.height;

    double offsetX = (size.width - renderedWidth) / 2;
    double offsetY = (size.height - renderedHeight) / 2;

    for (var rec in recognitions) {
      // Transformation: Coordonnée Réelle -> Coordonnée Écran
      double left = (rec.x1 * scaleX) + offsetX;
      double top = (rec.y1 * scaleY) + offsetY;
      double right = (rec.x2 * scaleX) + offsetX;
      double bottom = (rec.y2 * scaleY) + offsetY;

      // Dessin du rectangle
      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), boxPaint);
      
      // Affichage du score
      TextSpan span = TextSpan(
        style: TextStyle(color: Colors.white, backgroundColor: Colors.red, fontSize: 10),
        text: "${(rec.score * 100).toInt()}%",
      );
      TextPainter tp = TextPainter(text: span, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(left, top - 15));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}