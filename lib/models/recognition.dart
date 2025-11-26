import 'dart:io';
import 'dart:math';

class Recognition {
  // Coordonnées du rectangle englobant (AABB) pour le tri et l'OCR
  final double x1, y1, x2, y2;
  final double score;
  
  // NOUVEAU : Les 4 points précis du polygone orienté (OBB)
  // [Pt1_x, Pt1_y, Pt2_x, Pt2_y, ..., Pt4_y]
  final List<Point<double>> renderPoints; 
  final double angle; // Utile pour debug

  String text;
  File? cropFile;
  
  bool? isMisplaced;
  bool isInvalid;
  String? placementSuggestion;
  
  Recognition(
    this.x1, this.y1, this.x2, this.y2, this.score, {
    required this.renderPoints,
    this.angle = 0.0,
    this.text = "",
    this.cropFile,
    this.isMisplaced,
    this.isInvalid = false,
  });
}