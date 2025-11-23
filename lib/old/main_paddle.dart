import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

// Import du service Paddle
import 'paddle_ocr_service.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    const MaterialApp(debugShowCheckedModeBanner: false, home: YoloPage()),
  );
}

// Classe pour stocker les résultats de détection de cotes (YOLO)
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

  // --- REMPLACEMENT : Service Paddle ---
  final PaddleOcrService _ocrService = PaddleOcrService();

  // Contrôleur pour le zoom
  final TransformationController _transformController = TransformationController();
  double _currentScale = 1.0; 

  @override
  void initState() {
    super.initState();
    _loadModel(); // Charge YOLO
    _ocrService.initialize(); // Charge Paddle OCR

    // On écoute le changement de zoom
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
    _ocrService.dispose(); // Nettoyage Paddle
    _transformController.dispose();
    super.dispose();
  }

  // Chargement du modèle TFLite (YOLO)
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset(
        'assets/models/book_label_obb_s2_float32.tflite',
      );
      setState(() => _status = "Modèle YOLO prêt");
    } catch (e) {
      print("Erreur modèle YOLO: $e");
    }
  }

  // Sélection d'image
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
        // Paddle gère l'orientation, mais pour YOLO on aide un peu
        final oriented = img.bakeOrientation(decoded);
        _runInference(oriented);
      }
    }
  }

  // Exécution de l'inférence YOLO (Détection des boîtes)
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

    // 4. Decoding YOLO Output
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

    // Nettoyage et Tri
    List<Recognition> nmsRecognitions = _nms(rawRecognitions);
    List<Recognition> sortedRecognitions = _sortByReadingOrder(nmsRecognitions);

    setState(() {
      _recognitions = sortedRecognitions;
      _status = "${sortedRecognitions.length} cotes détectées. Lecture PaddleOCR...";
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

  // OCR avec Paddle (sur les crops YOLO)
  Future<void> _performOCR(
    img.Image originalImage,
    List<Recognition> boxes,
  ) async {
    final tempDir = await getTemporaryDirectory();

    for (var i = 0; i < boxes.length; i++) {
      var box = boxes[i];

      // --- DECOUPE DU CROP (Inchangé car nécessaire pour YOLO -> Paddle) ---
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

      // --- PRE-TRAITEMENT IMAGE (Gardé car aide l'OCR sur des petits crops) ---
      if (processed.height > processed.width * 1.2) {
        int newWidth = (processed.height * 0.8).toInt();
        processed = img.copyResize(processed, width: newWidth, height: processed.height, interpolation: img.Interpolation.cubic);
      }
      if (processed.height < 300) {
        processed = img.copyResize(processed, height: 300, interpolation: img.Interpolation.cubic);
      }
      processed = img.grayscale(processed);
      processed = img.contrast(processed, contrast: 150);

      // Sauvegarde fichier temporaire pour Paddle
      File cropFile = File('${tempDir.path}/temp_crop_$i.jpg');
      await cropFile.writeAsBytes(img.encodeJpg(processed, quality: 100));
      box.cropFile = cropFile;

      // --- APPEL A PADDLE OCR ---
      try {
        // On appelle le service sur la vignette
        final paddleResults = await _ocrService.processImage(cropFile);

        // Paddle peut renvoyer plusieurs bouts de texte dans la vignette
        // On les assemble (ex: "340" + "DRO" -> "340 DRO")
        String rawText = paddleResults.map((r) => r.text).join(" ");
        
        // --- NETTOYAGE REGEX DEWEY (Ta logique existante conservée) ---
        String cleanText = rawText.replaceAll("\n", " ").trim().toUpperCase();

        if (cleanText.isNotEmpty) {
          cleanText = cleanText.replaceAllMapped(
            RegExp(r'([OQDZSBILG])(?=\d)'),
            (Match m) {
              String letter = m.group(1)!;
              switch (letter) {
                case 'O': case 'Q': case 'D': return '0';
                case 'I': case 'L': return '1';
                case 'Z': return '2';
                case 'S': return '5';
                case 'G': return '6';
                case 'B': return '8';
                default: return letter;
              }
            },
          );
        }

        cleanText = cleanText.replaceAll(RegExp(r'[-_,]'), ".");
        cleanText = cleanText.replaceAll(RegExp(r'\s+\.'), ".");

        cleanText = cleanText.replaceAllMapped(
          RegExp(r'(\d{3})\s*(\d+)'),
          (Match m) => "${m[1]}.${m[2]}",
        );

        cleanText = cleanText.replaceAllMapped(
          RegExp(r'([0-9.]+)\s*([A-Z]+)'),
          (Match m) => "${m[1]}\n${m[2]}",
        );

        box.text = cleanText.isEmpty ? "?" : cleanText;
        print("Paddle Result #$i: ${box.text}");

      } catch (e) {
        print("Erreur Paddle sur box #$i: $e");
        box.text = "⚠️";
      }

      setState(() {}); // Mise à jour UI progressive
    }

    _checkShelfOrder();
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
    boxes.sort((a, b) => a.y1.compareTo(b.y1));

    List<List<Recognition>> rows = [];
    List<Recognition> currentRow = [];
    double avgHeight = boxes.map((e) => e.y2 - e.y1).reduce((a, b) => a + b) / boxes.length;
    double rowTolerance = avgHeight * 1.2;

    for (var box in boxes) {
      if (currentRow.isEmpty) {
        currentRow.add(box);
      } else {
        double yDiff = (box.y1 - currentRow.first.y1).abs();
        if (yDiff < rowTolerance) {
          currentRow.add(box);
        } else {
          currentRow.sort((a, b) => a.x1.compareTo(b.x1));
          rows.add(currentRow);
          currentRow = [box];
        }
      }
    }
    if (currentRow.isNotEmpty) {
      currentRow.sort((a, b) => a.x1.compareTo(b.x1));
      rows.add(currentRow);
    }
    return rows.expand((element) => element).toList();
  }

  void _checkShelfOrder() {
    if (_recognitions.isEmpty) return;
    List<_DeweyItem?> items = _recognitions.map((r) => _parseDewey(r.text)).toList();

    for (var r in _recognitions) r.isMisplaced = true;
    List<int> validIndices = [];

    for (var i = 0; i < items.length; i++) {
      var current = items[i];
      if (current == null) {
        _recognitions[i].isMisplaced = null;
        continue;
      }
      if (validIndices.isEmpty) {
        validIndices.add(i);
        _recognitions[i].isMisplaced = false;
        continue;
      }
      int lastIndex = validIndices.last;
      var lastValid = items[lastIndex]!;

      if (current.compareTo(lastValid) >= 0) {
        validIndices.add(i);
        _recognitions[i].isMisplaced = false;
      } else {
        bool intruderFound = false;
        if (validIndices.length >= 2) {
          int grandParentIndex = validIndices[validIndices.length - 2];
          var grandParent = items[grandParentIndex]!;
          if (current.compareTo(grandParent) >= 0) {
            _recognitions[lastIndex].isMisplaced = true;
            validIndices.removeLast();
            validIndices.add(i);
            _recognitions[i].isMisplaced = false;
            intruderFound = true;
          }
        }
        if (!intruderFound) {
          _recognitions[i].isMisplaced = true;
        }
      }
    }
    setState(() {});
  }

  _DeweyItem? _parseDewey(String text) {
    String clean = text.replaceAll("\n", " ").trim();
    final reg = RegExp(r'^([0-9.]+)\s*(.*)$');
    final match = reg.firstMatch(clean);
    if (match == null) return null;
    try {
      double number = double.parse(match.group(1)!);
      String rawSuffix = match.group(2) ?? "";
      List<String> parts = rawSuffix.trim().split(RegExp(r'\s+'));
      String p1 = parts.isNotEmpty ? parts[0] : "";
      String p2 = parts.length > 1 ? parts[1] : "";
      return _DeweyItem(number, p1, p2);
    } catch (e) {
      return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Scanner de Cotes (Paddle)"),
        backgroundColor: Colors.indigo,
        foregroundColor: Colors.white,
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
                          Text("Aucune image sélectionnée", style: TextStyle(color: Colors.grey)),
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
                                      imageSize: Size(_originalImage!.width.toDouble(), _originalImage!.height.toDouble()),
                                      widgetSize: Size(constraints.maxWidth, constraints.maxHeight),
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
          Expanded(
            flex: 50,
            child: Column(
              children: [
                if (_isBusy) const LinearProgressIndicator(color: Colors.indigo),
                Padding(
                  padding: const EdgeInsets.all(8.0),
                  child: Text(_status, style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.indigo)),
                ),
                Expanded(
                  child: _recognitions.isEmpty
                      ? const Center(child: Text("Les résultats s'afficheront ici."))
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
                                        border: Border.all(color: Colors.indigo.shade100),
                                        borderRadius: BorderRadius.circular(8),
                                      ),
                                      child: rec.cropFile != null
                                          ? ClipRRect(
                                              borderRadius: BorderRadius.circular(8),
                                              child: Image.file(rec.cropFile!, fit: BoxFit.contain),
                                            )
                                          : const Center(child: CircularProgressIndicator(strokeWidth: 2)),
                                    ),
                                    const SizedBox(width: 15),
                                    Expanded(
                                      child: Column(
                                        crossAxisAlignment: CrossAxisAlignment.start,
                                        children: [
                                          Container(
                                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                                            decoration: BoxDecoration(color: Colors.indigo, borderRadius: BorderRadius.circular(12)),
                                            child: Text("Cote #${index + 1}", style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold)),
                                          ),
                                          const SizedBox(height: 5),
                                          Text(rec.text.isEmpty ? "Analyse..." : rec.text, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                                          const SizedBox(height: 5),
                                          Text("Score: ${(rec.score * 100).toInt()}%", style: TextStyle(color: Colors.grey[600], fontSize: 12)),
                                        ],
                                      ),
                                    ),
                                    if (rec.isMisplaced == true)
                                      const Icon(Icons.warning_amber_rounded, color: Colors.red, size: 30)
                                    else if (rec.isMisplaced == false)
                                      const Icon(Icons.check_circle, color: Colors.green, size: 30)
                                    else
                                      const Icon(Icons.help_outline, color: Colors.grey),
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
                  onPressed: _isBusy ? null : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.camera_alt),
                  label: const Text("Caméra"),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.indigo, foregroundColor: Colors.white),
                ),
                ElevatedButton.icon(
                  onPressed: _isBusy ? null : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library),
                  label: const Text("Galerie"),
                  style: ElevatedButton.styleFrom(backgroundColor: Colors.white, foregroundColor: Colors.indigo),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DeweyItem implements Comparable<_DeweyItem> {
  final double number;
  final String part1;
  final String part2;
  _DeweyItem(this.number, this.part1, this.part2);

  @override
  int compareTo(_DeweyItem other) {
    int numComp = number.compareTo(other.number);
    if (numComp != 0) return numComp;
    int p1Comp = part1.compareTo(other.part1);
    if (p1Comp != 0) return p1Comp;
    return part2.compareTo(other.part2);
  }
}

class OcrPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final Size widgetSize;
  final double scale;

  OcrPainter({required this.recognitions, required this.imageSize, required this.widgetSize, required this.scale});

  @override
  void paint(Canvas canvas, Size size) {
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

    double strokeWidth = 3.0 / scale;
    double fontSize = 14.0 / scale;
    if (strokeWidth < 1.0) strokeWidth = 1.0;
    if (fontSize < 8.0) fontSize = 8.0;

    for (var i = 0; i < recognitions.length; i++) {
      var rec = recognitions[i];
      Color boxColor;
      if (rec.isMisplaced == true) {
        boxColor = Colors.red;
      } else if (rec.isMisplaced == false) {
        boxColor = Colors.greenAccent;
      } else {
        boxColor = Colors.blueAccent;
      }

      final Paint boxPaint = Paint()
        ..color = boxColor
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth;

      double left = (rec.x1 * scaleX) + offsetX;
      double top = (rec.y1 * scaleY) + offsetY;
      double right = (rec.x2 * scaleX) + offsetX;
      double bottom = (rec.y2 * scaleY) + offsetY;

      canvas.drawRect(Rect.fromLTRB(left, top, right, bottom), boxPaint);

      TextSpan span = TextSpan(
        style: TextStyle(
          color: boxColor,
          fontSize: fontSize,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(offset: Offset(-1 / scale, -1 / scale), color: Colors.black),
            Shadow(offset: Offset(1 / scale, -1 / scale), color: Colors.black),
            Shadow(offset: Offset(1 / scale, 1 / scale), color: Colors.black),
            Shadow(offset: Offset(-1 / scale, 1 / scale), color: Colors.black),
          ],
        ),
        text: "${i + 1}",
      );

      TextPainter tp = TextPainter(text: span, textDirection: TextDirection.ltr);
      tp.layout();
      tp.paint(canvas, Offset(left, top - (fontSize + 2)));
    }
  }

  @override
  bool shouldRepaint(covariant OcrPainter oldDelegate) {
    return oldDelegate.recognitions != recognitions || oldDelegate.scale != scale;
  }
}