import 'package:flutter/material.dart';
import 'pages/yolo_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(
    const MaterialApp(
      debugShowCheckedModeBanner: false, 
      home: YoloPage()
    ),
  );
}