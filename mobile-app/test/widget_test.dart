import 'package:flutter_test/flutter_test.dart';

import 'package:atlas_mobile/main.dart';

void main() {
  testWidgets('App starts on the start screen', (WidgetTester tester) async {
    await tester.pumpWidget(const AtlasApp());

    expect(find.text('ATLAS'), findsOneWidget);
    expect(find.textContaining('Start page'), findsOneWidget);
  });
}
