import 'package:material_ui/material_ui.dart';

void main() {
  runApp(const AtlasApp());
}

class AtlasApp extends StatelessWidget {
  const AtlasApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ATLAS',
      theme: ThemeData(colorSchemeSeed: Colors.indigo),
      home: const StartScreen(),
    );
  }
}

/// Placeholder start screen. Onboarding, permissions and Driving Mode are added
/// in Sprint 5 (HLF-7).
class StartScreen extends StatelessWidget {
  const StartScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ATLAS')),
      body: const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            'ATLAS driver app\nStart page',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}
