import 'package:flutter/material.dart';
import '../models/recognition.dart';

class OcrPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final Size widgetSize;
  final double scale;

  OcrPainter({
    required this.recognitions,
    required this.imageSize,
    required this.widgetSize,
    required this.scale,
  });

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
      // --- LOGIQUE DE COULEUR MISE A JOUR ---
      Color boxColor;

      if (rec.isInvalid) {
        boxColor =
            Colors.orange; // <--- Format incorrect (Vérification humaine)
      } else if (rec.isMisplaced == true) {
        boxColor = Colors.red; // <--- Erreur de tri
      } else if (rec.isMisplaced == false) {
        boxColor = Colors.greenAccent; // <--- Bon tri
      } else {
        boxColor = Colors.blueAccent; // <--- En cours / Illisible
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

      TextPainter tp = TextPainter(
        text: span,
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, Offset(left, top - (fontSize + 2)));
    }
  }

  @override
  bool shouldRepaint(covariant OcrPainter oldDelegate) {
    return oldDelegate.recognitions != recognitions ||
        oldDelegate.scale != scale;
  }
}
