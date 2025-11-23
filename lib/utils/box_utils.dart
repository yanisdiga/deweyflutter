import 'dart:math';
import '../models/recognition.dart';

class BoxUtils {
  // Calcul de l'Intersection over Union
  static double calculateIoU(Recognition boxA, Recognition boxB) {
    double xA = max(boxA.x1, boxB.x1);
    double yA = max(boxA.y1, boxB.y1);
    double xB = min(boxA.x2, boxB.x2);
    double yB = min(boxA.y2, boxB.y2);
    double interArea = max(0, xB - xA) * max(0, yB - yA);
    double boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1);
    double boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1);
    return interArea / (boxAArea + boxBArea - interArea);
  }

  // Non-Maximum Suppression (Supprime les doublons de boîtes)
  static List<Recognition> nms(List<Recognition> boxes, double iouThreshold) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => b.score.compareTo(a.score));
    List<Recognition> selected = [];
    List<bool> active = List.filled(boxes.length, true);

    for (int i = 0; i < boxes.length; i++) {
      if (active[i]) {
        selected.add(boxes[i]);
        for (int j = i + 1; j < boxes.length; j++) {
          if (active[j]) {
            double iou = calculateIoU(boxes[i], boxes[j]);
            if (iou > iouThreshold) active[j] = false;
          }
        }
      }
    }
    return selected;
  }

  // Tri par ordre de lecture (Haut -> Bas, Gauche -> Droite)
  static List<Recognition> sortByReadingOrder(List<Recognition> boxes) {
    if (boxes.isEmpty) return [];
    boxes.sort((a, b) => a.y1.compareTo(b.y1));

    List<List<Recognition>> rows = [];
    List<Recognition> currentRow = [];
    
    if (boxes.isEmpty) return [];
    
    double avgHeight =
        boxes.map((e) => e.y2 - e.y1).reduce((a, b) => a + b) / boxes.length;
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
}