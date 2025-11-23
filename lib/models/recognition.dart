import 'dart:io';

class Recognition {
  final double x1, y1, x2, y2, score;
  String text;
  File? cropFile;
  
  bool? isMisplaced; // true=Rouge, false=Vert
  bool isInvalid;    // <--- NOUVEAU : true=Orange (Format incorrect)

  Recognition(
    this.x1,
    this.y1,
    this.x2,
    this.y2,
    this.score, {
    this.text = "",
    this.cropFile,
    this.isMisplaced,
    this.isInvalid = false, // Par défaut, on suppose que c'est valide
  });
}