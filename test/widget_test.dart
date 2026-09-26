import 'package:flutter_test/flutter_test.dart';
import 'package:translation_call_demo/main.dart';

void main() {
  testWidgets('App renders Home Screen smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const TranslationCallApp());
    expect(find.text('Translation Call Demo'), findsOneWidget);
    expect(find.text('User A (Urdu Speaker)'), findsOneWidget);
  });
}
