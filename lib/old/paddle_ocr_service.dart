import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
// --- AJOUT IMPORTANT ICI ---
import 'package:flutter/material.dart'; 
// ou import 'dart:ui'; 
// C'est nécessaire pour que le type 'Rect' soit reconnu
// ---------------------------

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';

// --- Types de retour ---
class OcrResult {
  final String text;
  final Rect rect; // Position dans l'image (nécessite l'import flutter/material.dart)
  final double confidence; 

  OcrResult(this.text, this.rect, {this.confidence = 1.0});
}

class _SimpleBox {
  final int x, y, width, height;
  _SimpleBox(this.x, this.y, this.width, this.height);
}

class PaddleOcrService {
  // Singleton
  static final PaddleOcrService _instance = PaddleOcrService._internal();
  factory PaddleOcrService() => _instance;
  PaddleOcrService._internal();

  // Sessions & Config
  OrtSession? _sessionDet;
  OrtSession? _sessionRec;
  String? _detInputName;
  String? _recInputName;
  List<String> _vocab = [];
  bool isLoaded = false;

  // Configuration du modèle
  static const double DET_THRESHOLD = 0.3;
  static const double UNCLIP_RATIO = 1.8; 
  static const int REC_IMG_H = 48; 

  /// Initialisation
  Future<void> initialize() async {
    if (isLoaded) return;

    try {
      OrtEnv.instance.init();
      
      // 1. Charger le vocabulaire
      final vocabStr = await rootBundle.loadString('assets/en_dict.txt');
      final lines = vocabStr.split('\n').map((e) => e.trim()).toList();
      _vocab = ["<blank>", ...lines, " "];

      // 2. Copier les modèles depuis assets vers dossier temp
      final detPath = await _copyAssetToLocal('assets/models/det_v4_ch.onnx');
      final recPath = await _copyAssetToLocal('assets/models/rec_v4_en.onnx');

      // 3. Créer les sessions
      final sessionOptions = OrtSessionOptions();
      try {
        _sessionDet = OrtSession.fromFile(File(detPath), sessionOptions);
        _detInputName = _sessionDet!.inputNames.first;

        _sessionRec = OrtSession.fromFile(File(recPath), sessionOptions);
        _recInputName = _sessionRec!.inputNames.first;
      } finally {
        sessionOptions.release();
      }

      isLoaded = true;
      print("✅ PaddleOCR Service Initialisé");
    } catch (e) {
      print("❌ Erreur init PaddleOCR: $e");
      rethrow;
    }
  }

  /// Méthode principale
  Future<List<OcrResult>> processImage(File imageFile) async {
    if (!isLoaded) await initialize();

    final bytes = await imageFile.readAsBytes();
    final originalImage = img.decodeImage(bytes);
    if (originalImage == null) throw Exception("Impossible de décoder l'image");

    final boxes = await _runDetection(originalImage);

    List<OcrResult> results = [];

    for (var box in boxes) {
      int x1 = max(0, box.x);
      int y1 = max(0, box.y);
      int x2 = min(originalImage.width, box.x + box.width);
      int y2 = min(originalImage.height, box.y + box.height);

      if ((x2 - x1) < 4 || (y2 - y1) < 4) continue;

      final crop = img.copyCrop(originalImage, x: x1, y: y1, width: x2 - x1, height: y2 - y1);
      
      final text = await _runRecognition(crop);
      
      if (text.isNotEmpty) {
        results.add(OcrResult(
          text, 
          Rect.fromLTRB(x1.toDouble(), y1.toDouble(), x2.toDouble(), y2.toDouble())
        ));
      }
    }

    return results;
  }

  /// Libérer la mémoire
  void dispose() {
    _sessionDet?.release();
    _sessionRec?.release();
    OrtEnv.instance.release();
    isLoaded = false;
  }

  // -----------------------------------------------------------------------
  // --- PRIVATE: DETECTION LOGIC ---
  // -----------------------------------------------------------------------
  Future<List<_SimpleBox>> _runDetection(img.Image image) async {
    int h = image.height;
    int w = image.width;
    int limit = 960;
    double ratio = 1.0;
    if (max(h, w) > limit) ratio = limit / max(h, w);
    
    int resizeH = (h * ratio / 32).round() * 32;
    int resizeW = (w * ratio / 32).round() * 32;
    if (resizeH < 32) resizeH = 32;
    if (resizeW < 32) resizeW = 32;

    final resized = img.copyResize(image, width: resizeW, height: resizeH);
    final floatList = _imageToFloatList(resized, mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]);

    final inputOrt = OrtValueTensor.createTensorWithDataList(floatList, [1, 3, resizeH, resizeW]);
    final runOptions = OrtRunOptions();
    
    final outputs = await _sessionDet!.run(runOptions, {_detInputName!: inputOrt});
    
    inputOrt.release();
    runOptions.release();

    final outputTensor = outputs[0]!.value as List<List<List<List<double>>>>;
    final map = outputTensor[0][0]; 

    for (var e in outputs) e?.release();

    return _findBoxesFromHeatmap(map, ratio, DET_THRESHOLD);
  }

  // -----------------------------------------------------------------------
  // --- PRIVATE: RECOGNITION LOGIC ---
  // -----------------------------------------------------------------------
  Future<String> _runRecognition(img.Image crop) async {
    double ratio = crop.width / crop.height;
    int resizeW = (REC_IMG_H * ratio).toInt();
    if (resizeW < 32) resizeW = 32;

    final resized = img.copyResize(crop, width: resizeW, height: REC_IMG_H);
    final floatList = _imageToFloatList(resized, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]);

    final inputOrt = OrtValueTensor.createTensorWithDataList(floatList, [1, 3, REC_IMG_H, resizeW]);
    final runOptions = OrtRunOptions();

    final outputs = await _sessionRec!.run(runOptions, {_recInputName!: inputOrt});
    
    inputOrt.release();
    runOptions.release();

    final preds = outputs[0]!.value as List<List<List<double>>>;
    final timeSteps = preds[0];

    for (var e in outputs) e?.release();

    return _decodeCTC(timeSteps);
  }

  // -----------------------------------------------------------------------
  // --- UTILS ---
  // -----------------------------------------------------------------------
  
  Float32List _imageToFloatList(img.Image image, {required List<double> mean, required List<double> std}) {
    final list = Float32List(1 * 3 * image.height * image.width);
    int ptr = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < image.height; y++) {
        for (int x = 0; x < image.width; x++) {
          final pixel = image.getPixel(x, y);
          double val = 0;
          if (c == 0) val = pixel.r / 255.0;
          if (c == 1) val = pixel.g / 255.0;
          if (c == 2) val = pixel.b / 255.0;
          list[ptr++] = (val - mean[c]) / std[c];
        }
      }
    }
    return list;
  }

  String _decodeCTC(List<List<double>> timeSteps) {
    String text = "";
    int lastIndex = -1;
    int blankIndex = 0; 

    for (var stepProbs in timeSteps) {
      int maxIdx = 0;
      double maxVal = stepProbs[0];
      for(int i=1; i<stepProbs.length; i++){
        if(stepProbs[i] > maxVal) { maxVal = stepProbs[i]; maxIdx = i; }
      }
      if (maxIdx != blankIndex && maxIdx != lastIndex) {
        if (maxIdx < _vocab.length) text += _vocab[maxIdx];
      }
      lastIndex = maxIdx;
    }
    return text;
  }

  List<_SimpleBox> _findBoxesFromHeatmap(List<List<double>> map, double ratio, double thresh) {
    List<_SimpleBox> boxes = [];
    int h = map.length;
    int w = map[0].length;
    var visited = List.generate(h, (_) => List.filled(w, false));

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (map[y][x] > thresh && !visited[y][x]) {
          var bounds = _floodFill(map, visited, x, y, w, h, thresh);
          if (bounds.width > 2 && bounds.height > 2) {
             double rawX = bounds.x / ratio;
             double rawY = bounds.y / ratio;
             double rawW = bounds.width / ratio;
             double rawH = bounds.height / ratio;

             double centerX = rawX + (rawW / 2);
             double centerY = rawY + (rawH / 2);
             double newW = rawW * UNCLIP_RATIO;
             double newH = rawH * UNCLIP_RATIO;

             boxes.add(_SimpleBox(
               (centerX - newW / 2).toInt(),
               (centerY - newH / 2).toInt(),
               newW.toInt(),
               newH.toInt()
             ));
          }
        }
      }
    }
    boxes.sort((a,b) => a.y.compareTo(b.y));
    return boxes;
  }

  _SimpleBox _floodFill(List<List<double>> map, List<List<bool>> visited, int sx, int sy, int w, int h, double thresh) {
    int minX = sx, maxX = sx, minY = sy, maxY = sy;
    List<Point> stack = [Point(sx, sy)];
    visited[sy][sx] = true;

    while(stack.isNotEmpty){
      final p = stack.removeLast();
      if(p.x < minX) minX = p.x.toInt();
      if(p.x > maxX) maxX = p.x.toInt();
      if(p.y < minY) minY = p.y.toInt();
      if(p.y > maxY) maxY = p.y.toInt();

      const dirs = [[0,1], [0,-1], [1,0], [-1,0]];
      for(var d in dirs){
        int nx = p.x.toInt() + d[0];
        int ny = p.y.toInt() + d[1];
        if(nx >= 0 && nx < w && ny >= 0 && ny < h){
          if(!visited[ny][nx] && map[ny][nx] > thresh){
            visited[ny][nx] = true;
            stack.add(Point(nx, ny));
          }
        }
      }
    }
    return _SimpleBox(minX, minY, maxX - minX, maxY - minY);
  }

  Future<String> _copyAssetToLocal(String assetName) async {
    final byteData = await rootBundle.load(assetName);
    final file = File('${(await getApplicationDocumentsDirectory()).path}/${assetName.split('/').last}');
    await file.writeAsBytes(byteData.buffer.asUint8List());
    return file.path;
  }
}