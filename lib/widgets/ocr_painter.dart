import 'package:flutter/material.dart';
import '../models/recognition.dart';
import 'dart:math'; // Nécessaire pour min()

class OcrPainter extends CustomPainter {
  final List<Recognition> recognitions;
  final Size imageSize;
  final Size widgetSize;
  final double scale;

  // 1. AJOUT DU PARAMÈTRE
  final bool showArrows;

  OcrPainter({
    required this.recognitions,
    required this.imageSize,
    required this.widgetSize,
    required this.scale,
    this.showArrows = true,
  });

  @override
  void paint(Canvas canvas, Size size) {
    // -----------------------------------------------------------
    // 1. CALCULS DE MISE A L'ECHELLE
    // -----------------------------------------------------------
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
    strokeWidth = strokeWidth.clamp(1.5, 5.0);
    fontSize = fontSize.clamp(10.0, 30.0);

    // -----------------------------------------------------------
    // 2. DESSIN DES CADRES ET NUMÉROS
    // -----------------------------------------------------------
    for (var i = 0; i < recognitions.length; i++) {
      var rec = recognitions[i];

      Color boxColor;
      if (rec.isInvalid) {
        boxColor = Colors.orange;
      } else if (rec.isMisplaced == true) {
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
          shadows: const [
            Shadow(offset: Offset(-1, -1), color: Colors.black),
            Shadow(offset: Offset(1, -1), color: Colors.black),
            Shadow(offset: Offset(1, 1), color: Colors.black),
            Shadow(offset: Offset(-1, 1), color: Colors.black),
          ],
        ),
        text: "${i + 1}",
      );

      TextPainter tp = TextPainter(
        text: span,
        textDirection: TextDirection.ltr,
      );
      tp.layout();
      tp.paint(canvas, Offset(left, top - (fontSize + 4)));
    }

    // -----------------------------------------------------------
    // 3. DESSIN DES LIENS (CONNECTIONS ARRONDIS)
    // -----------------------------------------------------------
    if (showArrows) {
      for (var i = 0; i < recognitions.length; i++) {
        var rec = recognitions[i];

        if (rec.isMisplaced == true && rec.placementSuggestion != null) {
          _drawConnectionLink(
            canvas,
            rec,
            recognitions,
            scaleX,
            scaleY,
            offsetX,
            offsetY,
            strokeWidth,
            i, // On passe l'index pour varier les couleurs/hauteurs
          );
        }
      }
    }
  }

  /// Dessine un lien style "Agrafe" avec COINS ARRONDIS par le HAUT
  void _drawConnectionLink(
    Canvas canvas,
    Recognition intruder,
    List<Recognition> allRecs,
    double scaleX,
    double scaleY,
    double offsetX,
    double offsetY,
    double baseStrokeWidth,
    int index, // Index de l'élément courant
  ) {
    // ============================================================
    // 🛠️ ZONE DE CONFIGURATION DYNAMIQUE
    // ============================================================

    // 1. Parsing
    String suggestion = intruder.placementSuggestion!;
    final match = RegExp(r'#(\d+)').firstMatch(suggestion);
    if (match == null) return;

    int targetNum = int.parse(match.group(1)!);
    int targetIndex = targetNum - 1;

    if (targetIndex < 0 || targetIndex >= allRecs.length) return;
    Recognition target = allRecs[targetIndex];

    // 2. Coordonnées
    double startX = ((intruder.x1 + intruder.x2) / 2 * scaleX) + offsetX;
    double startY = (intruder.y1 * scaleY) + offsetY;

    bool isBefore = suggestion.contains("Avant");
    double targetXRaw = isBefore ? target.x1 : target.x2;
    double targetX = (targetXRaw * scaleX) + offsetX;
    double targetY = (target.y1 * scaleY) + offsetY;

    // 3. CONFIGURATION VISUELLE (COULEUR & HAUTEUR)

    // A. Couleur Unique (DeepPurple pour être visible mais sobre)
    final Color arrowColor = Colors.red;
    final double lineThickness = baseStrokeWidth * 1.0;

    // B. Hauteur basée sur la distance horizontale
    // Plus c'est loin, plus c'est haut.
    double distanceHorizontale = (targetX - startX).abs();

    // Formule : Base 30 + 15% de la distance, plafonné à 120
    final double archHeight = (30.0 + distanceHorizontale * 0.05).clamp(
      30.0,
      60.0,
    );

    // Rayon des coins
    double cornerRadius = 15.0;

    // 4. Peinture
    Paint linkPaint = Paint()
      ..color = arrowColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = lineThickness
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;

    // 5. Calcul du chemin arrondi
    double topY = startY - archHeight;
    double direction = (targetX > startX) ? 1.0 : -1.0;
    double safeRadius = min(cornerRadius, distanceHorizontale / 2);

    Path path = Path();
    path.moveTo(startX, startY);

    // Montée
    path.lineTo(startX, topY + safeRadius);

    // Virage 1
    path.quadraticBezierTo(
      startX,
      topY,
      startX + (safeRadius * direction),
      topY,
    );

    // Traversée
    path.lineTo(targetX - (safeRadius * direction), topY);

    // Virage 2
    path.quadraticBezierTo(targetX, topY, targetX, topY + safeRadius);

    // Descente
    path.lineTo(targetX, targetY - 10);

    // Dessin
    canvas.drawPath(path, linkPaint);

    // Pointe
    _drawArrowHeadDown(canvas, Offset(targetX, targetY - 10), arrowColor);
  }

  void _drawArrowHeadDown(Canvas canvas, Offset tip, Color color) {
    Paint arrowPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    // ---------------------------------------------------------
    // 🛠️ RÉGLAGE DE LA TAILLE
    // C'était 8.0 avant. Essayez 5.0 (moyen) ou 4.0 (petit)
    // ---------------------------------------------------------
    double arrowSize = 5.0;

    Path head = Path();
    head.moveTo(tip.dx, tip.dy + arrowSize);
    head.lineTo(tip.dx - arrowSize, tip.dy - arrowSize);
    head.lineTo(tip.dx + arrowSize, tip.dy - arrowSize);
    head.close();

    // Petite ombre noire pour la pointe aussi
    Paint shadowPaint = Paint()
      ..color = Colors.black
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;

    canvas.drawPath(head, arrowPaint);
    canvas.drawPath(head, shadowPaint);
  }

  @override
  bool shouldRepaint(covariant OcrPainter oldDelegate) {
    return oldDelegate.recognitions != recognitions ||
        oldDelegate.scale != scale ||
        oldDelegate.showArrows != showArrows;
  }
}
