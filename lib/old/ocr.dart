import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: const OcrScreen());
  }
}

class OcrScreen extends StatefulWidget {
  const OcrScreen({super.key});
  @override
  State<OcrScreen> createState() => _OcrScreenState();
}

class _OcrScreenState extends State<OcrScreen> {
  // Chemins et Sessions
  OrtSession? _sessionDet;
  OrtSession? _sessionRec;
  String? _detInputName;
  String? _recInputName;
  List<String> _vocab = [];

  File? _imageFile;
  img.Image? _displayImage; // Image pour affichage
  String _status = "Prêt. Chargez une image.";
  List<OcrResult> _results = [];

  // --- CONFIGURATION PADDING (Comme votre Python) ---
  // final int padTop = 10;
  // final int padBottom = 0;
  // final int padX = 15;
  final double addRatio = 2.0; // Même valeur que dans le Python

  @override
  void initState() {
    super.initState();
    _initEnvironment();
  }

  Future<void> _initEnvironment() async {
    try {
      OrtEnv.instance.init();
      setState(() => _status = "Chargement des modèles...");

      // 1. Charger Vocabulaire
      final vocabStr = await rootBundle.loadString('assets/en_dict.txt');
      final lines = vocabStr.split('\n').map((e) => e.trim()).toList();
      // Ajout du <blank> à l'index 0 et espace à la fin (comme Python)
      _vocab = ["<blank>", ...lines, " "];

      // 2. Charger Modèles ONNX depuis les assets vers un dossier temporaire
      final detPath = await _copyAssetToLocal('assets/models/det_v4_ch.onnx');
      final recPath = await _copyAssetToLocal('assets/models/rec_v4_en.onnx');

      // 3. Créer les sessions avec les options obligatoires
      final sessionOptions = OrtSessionOptions();
      try {
        _sessionDet = OrtSession.fromFile(File(detPath), sessionOptions);
        // --- RECUPERER LE NOM ENTREE DET ---
        _detInputName = _sessionDet!.inputNames.first;

        _sessionRec = OrtSession.fromFile(File(recPath), sessionOptions);
        // --- RECUPERER LE NOM ENTREE REC ---
        _recInputName = _sessionRec!.inputNames.first;
      } finally {
        sessionOptions.release();
      }

      setState(() => _status = "Modèles chargés !");
    } catch (e) {
      setState(() => _status = "Erreur init: $e");
    }
  }

  // Utilitaire pour copier l'asset dans le stockage accessible par le C++ ONNX
  Future<String> _copyAssetToLocal(String assetName) async {
    final byteData = await rootBundle.load(assetName);
    final file = File(
      '${(await getApplicationDocumentsDirectory()).path}/${assetName.split('/').last}',
    );
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _imageFile = File(pickedFile.path);
        _status = "Traitement en cours...";
        _results.clear();
      });

      // Décoder l'image pour le traitement
      final bytes = await _imageFile!.readAsBytes();
      final decodedImg = img.decodeImage(bytes);

      if (decodedImg != null) {
        _runPipeline(decodedImg);
      }
    }
  }

  // --- PIPELINE PRINCIPAL ---
  Future<void> _runPipeline(img.Image originalImage) async {
    if (_sessionDet == null || _sessionRec == null) return;

    try {
      // 1. Detection
      final boxes = await _runDetection(originalImage);

      List<OcrResult> localResults = [];

      // 2. Reconnaissance sur chaque boite
      for (var box in boxes) {
        // Plus de calcul de padding ici, la box est déjà à la bonne taille

        // Sécurité pour ne pas sortir de l'image (Clamping)
        int x1 = max(0, box.x);
        int y1 = max(0, box.y);
        int x2 = min(originalImage.width, box.x + box.width);
        int y2 = min(originalImage.height, box.y + box.height);

        // Vérification taille minimale
        if ((x2 - x1) < 5 || (y2 - y1) < 5) continue;

        // Crop
        final crop = img.copyCrop(
          originalImage,
          x: x1,
          y: y1,
          width: x2 - x1,
          height: y2 - y1,
        );

        // OCR
        final text = await _runRecognition(crop);

        // On stocke le résultat
        localResults.add(
          OcrResult(
            Rect.fromLTRB(
              x1.toDouble(),
              y1.toDouble(),
              x2.toDouble(),
              y2.toDouble(),
            ),
            text,
          ),
        );
      }

      setState(() {
        _results = localResults;
        _displayImage =
            originalImage; // On garde l'originale pour dessiner par dessus
        _status = "Terminé : ${_results.length} textes trouvés.";
      });
    } catch (e) {
      setState(() => _status = "Erreur pipeline: $e");
      print(e);
    }
  }

  // --- 1. DETECTION (Simplified) ---
  Future<List<SimpleBox>> _runDetection(img.Image image) async {
    // Resize logique (Multiple de 32)
    int h = image.height;
    int w = image.width;
    int limit = 960;
    double ratio = 1.0;
    if (max(h, w) > limit) {
      ratio = limit / max(h, w);
    }
    int resizeH = (h * ratio / 32).round() * 32;
    int resizeW = (w * ratio / 32).round() * 32;
    if (resizeH < 32) resizeH = 32;
    if (resizeW < 32) resizeW = 32;

    final resized = img.copyResize(image, width: resizeW, height: resizeH);

    // Preprocess: Normalize & HWC -> CHW
    // Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225]
    final floatList = Float32List(1 * 3 * resizeH * resizeW);

    int ptr = 0;
    // Planar layout (RRR...GGG...BBB...)
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < resizeH; y++) {
        for (int x = 0; x < resizeW; x++) {
          final pixel = resized.getPixel(x, y);
          double val = 0;
          double mean = 0;
          double std = 0;

          if (c == 0) {
            val = pixel.r / 255.0;
            mean = 0.485;
            std = 0.229;
          }
          if (c == 1) {
            val = pixel.g / 255.0;
            mean = 0.456;
            std = 0.224;
          }
          if (c == 2) {
            val = pixel.b / 255.0;
            mean = 0.406;
            std = 0.225;
          }

          floatList[ptr++] = (val - mean) / std;
        }
      }
    }

    final inputOrt = OrtValueTensor.createTensorWithDataList(floatList, [
      1,
      3,
      resizeH,
      resizeW,
    ]);

    final runOptions = OrtRunOptions();
    final outputs = await _sessionDet!.run(runOptions, {
      _detInputName!: inputOrt,
    });

    // Libération immédiate de l'input et des options de run
    inputOrt.release();
    runOptions.release();

    // Output shape: [1, 1, H, W] -> Heatmap
    // Extraction des données AVANT de libérer les outputs
    final outputTensor = outputs[0]!.value as List<List<List<List<double>>>>;
    final map = outputTensor[0][0]; // H x W map

    // CORRECTION : Libération des outputs un par un
    for (var element in outputs) {
      element?.release();
    }

    // Post-Process
    return _findBoxesFromHeatmap(map, ratio, 0.3);
  }

  // --- 2. RECONNAISSANCE (Paddle V4) ---
  Future<String> _runRecognition(img.Image crop) async {
    // Resize fix height 48, keep ratio
    int inputH = 48;
    double ratio = crop.width / crop.height;
    int resizeW = (inputH * ratio).toInt();
    if (resizeW < 32) resizeW = 32;

    final resized = img.copyResize(crop, width: resizeW, height: inputH);

    // Normalization: (val - 0.5) / 0.5
    final floatList = Float32List(1 * 3 * inputH * resizeW);
    int ptr = 0;

    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < inputH; y++) {
        for (int x = 0; x < resizeW; x++) {
          final pixel = resized.getPixel(x, y);
          double val = 0;
          if (c == 0) val = pixel.r / 255.0;
          if (c == 1) val = pixel.g / 255.0;
          if (c == 2) val = pixel.b / 255.0;

          floatList[ptr++] = (val - 0.5) / 0.5;
        }
      }
    }

    final inputOrt = OrtValueTensor.createTensorWithDataList(floatList, [
      1,
      3,
      inputH,
      resizeW,
    ]);

    final runOptions = OrtRunOptions();
    final outputs = await _sessionRec!.run(runOptions, {
      _recInputName!: inputOrt,
    });

    // Libération inputs
    inputOrt.release();
    runOptions.release();

    // Output shape: [1, Time, VocabSize]
    final preds = outputs[0]!.value as List<List<List<double>>>;
    // On copie les données dont on a besoin car on va release juste après
    final timeSteps = preds[0];

    // CORRECTION : Libération des outputs un par un
    for (var element in outputs) {
      element?.release();
    }

    // CTC Decode
    String text = "";
    int lastIndex = -1;
    int blankIndex = 0;

    for (var stepProbs in timeSteps) {
      // Argmax manual
      int maxIdx = 0;
      double maxVal = stepProbs[0];
      for (int i = 1; i < stepProbs.length; i++) {
        if (stepProbs[i] > maxVal) {
          maxVal = stepProbs[i];
          maxIdx = i;
        }
      }

      if (maxIdx != blankIndex && maxIdx != lastIndex) {
        if (maxIdx < _vocab.length) {
          text += _vocab[maxIdx];
        }
      }
      lastIndex = maxIdx;
    }

    return text;
  }

  // --- UTILITAIRE: Find Boxes (Simplifié sans OpenCV) ---
  // Cherche les zones connectées > threshold
  // C'est une version très naïve pour remplacer cv2.findContours
  List<SimpleBox> _findBoxesFromHeatmap(
    List<List<double>> map,
    double ratio,
    double thresh,
  ) {
    List<SimpleBox> boxes = [];
    int h = map.length;
    int w = map[0].length;

    // Matrice visitée
    var visited = List.generate(h, (_) => List.filled(w, false));

    // FACTEUR D'AGRANDISSEMENT (SCALE)
    // 1.0 = taille détectée (souvent trop fin)
    // 1.6 = standard PaddleOCR (recommandé)
    // 2.0 = très large (si vous ratez encore des morceaux)
    double unclipRatio = addRatio;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (map[y][x] > thresh && !visited[y][x]) {
          // 1. Trouver le rectangle serré (Bounding Box du squelette)
          var bounds = _floodFill(map, visited, x, y, w, h, thresh);

          if (bounds.width > 2 && bounds.height > 2) {
            // 2. Convertir en coordonnées réelles de l'image d'origine
            double rawX = bounds.x / ratio;
            double rawY = bounds.y / ratio;
            double rawW = bounds.width / ratio;
            double rawH = bounds.height / ratio;

            // 3. EXPANSION DEPUIS LE CENTRE (La méthode correcte)
            double centerX = rawX + (rawW / 2);
            double centerY = rawY + (rawH / 2);

            double newW = rawW * unclipRatio;
            double newH = rawH * unclipRatio;

            double newX = centerX - (newW / 2);
            double newY = centerY - (newH / 2);

            boxes.add(
              SimpleBox(newX.toInt(), newY.toInt(), newW.toInt(), newH.toInt()),
            );
          }
        }
      }
    }
    // Tri vertical (Haut -> Bas)
    boxes.sort((a, b) => a.y.compareTo(b.y));
    return boxes;
  }

  SimpleBox _floodFill(
    List<List<double>> map,
    List<List<bool>> visited,
    int startX,
    int startY,
    int w,
    int h,
    double thresh,
  ) {
    int minX = startX, maxX = startX;
    int minY = startY, maxY = startY;

    List<Point> stack = [Point(startX, startY)];
    visited[startY][startX] = true;

    while (stack.isNotEmpty) {
      final p = stack.removeLast();
      if (p.x < minX) minX = p.x.toInt();
      if (p.x > maxX) maxX = p.x.toInt();
      if (p.y < minY) minY = p.y.toInt();
      if (p.y > maxY) maxY = p.y.toInt();

      // Voisins 4 directions
      final dirs = [
        [0, 1],
        [0, -1],
        [1, 0],
        [-1, 0],
      ];
      for (var d in dirs) {
        int nx = p.x.toInt() + d[0];
        int ny = p.y.toInt() + d[1];
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          if (!visited[ny][nx] && map[ny][nx] > thresh) {
            visited[ny][nx] = true;
            stack.add(Point(nx, ny));
          }
        }
      }
    }
    return SimpleBox(minX, minY, maxX - minX, maxY - minY);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("PaddleOCR Flutter")),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              _status,
              style: const TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
          Expanded(
            child: _imageFile == null
                ? const Center(child: Text("Aucune image"))
                : Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(_imageFile!, fit: BoxFit.contain),
                      if (_results.isNotEmpty && _displayImage != null)
                        CustomPaint(
                          painter: OverlayPainter(
                            _results,
                            _displayImage!.width.toDouble(),
                            _displayImage!.height.toDouble(),
                          ),
                        ),
                    ],
                  ),
          ),
          ElevatedButton(
            onPressed: _pickImage,
            child: const Text("Importer Image"),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }
}

// --- Classes Helper ---

class SimpleBox {
  final int x, y, width, height;
  SimpleBox(this.x, this.y, this.width, this.height);
}

class OcrResult {
  final Rect rect;
  final String text;
  OcrResult(this.rect, this.text);
}

class BBoxPainter extends CustomPainter {
  final img.Image bgImage;
  final List<OcrResult> results;

  BBoxPainter(this.bgImage, this.results);

  @override
  void paint(Canvas canvas, Size size) {
    // Dessiner l'image ajustée à l'écran
    final uiImg = decodeImageFromList(img.encodeJpg(bgImage));
    // Note: C'est lent de re-encoder en jpg à chaque frame, optimiser pour prod
    // Pour cet exemple, on fait simple.
    // Dans une vraie app, convertissez img.Image en ui.Image une seule fois.
  }

  // Correction rapide pour le dessin :
  // CustomPainter avec image brute est complexe en Flutter pur.
  // On va simplifier : On affiche l'image standard et on dessine des rectangles relatifs.
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

class OverlayPainter extends CustomPainter {
  final List<OcrResult> results;
  final double imgW;
  final double imgH;

  OverlayPainter(this.results, this.imgW, this.imgH);

  @override
  void paint(Canvas canvas, Size size) {
    // Calcul du ratio d'affichage (BoxFit.contain)
    double scaleX = size.width / imgW;
    double scaleY = size.height / imgH;
    double scale = min(scaleX, scaleY);

    double offsetX = (size.width - imgW * scale) / 2;
    double offsetY = (size.height - imgH * scale) / 2;

    final paintBox = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final textStyle = const TextStyle(
      color: Colors.red,
      fontSize: 14,
      backgroundColor: Colors.white,
    );

    for (var res in results) {
      // Transformation des coordonnées image -> coordonnées écran
      final rect = Rect.fromLTWH(
        offsetX + res.rect.left * scale,
        offsetY + res.rect.top * scale,
        res.rect.width * scale,
        res.rect.height * scale,
      );

      canvas.drawRect(rect, paintBox);

      final textSpan = TextSpan(text: res.text, style: textStyle);
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(rect.left, rect.top - 20));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
