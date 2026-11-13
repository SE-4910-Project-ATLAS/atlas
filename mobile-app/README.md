# Mobile App

Driver mobile application for Android and iOS: onboarding, alert preferences, route selection, Driving Mode, voice alerts and trip summary (HLF-7).

**Jira epic:** S4PT-30 Mobile App

## Setup

The app is a Flutter project (Flutter 3.50.0, Dart 3.14.0). It is not part of Docker
Compose: it runs on a phone or an emulator and talks to the backend started with
`docker compose up` in the repository root.

With Flutter installed locally (https://docs.flutter.dev/get-started/install):

```bash
cd mobile-app
flutter pub get
flutter run          # on a connected phone or a running emulator
flutter test
```

On an Android emulator the backend on the host is reachable at `http://10.0.2.2:8000`;
on a physical phone use the IP address of the machine that runs Docker Compose.
The iOS app is built on macOS with Xcode (SR-11).

**Without a local Flutter installation**, the project can be checked and an Android debug
APK built in Docker. The first run downloads the Flutter SDK into a Docker volume:

```bash
docker run --rm -v atlas-flutter-3.50.0:/sdk -v "$PWD:/w" -w /w ghcr.io/cirruslabs/flutter:stable bash -c '
  [ -x /sdk/flutter/bin/flutter ] || curl -sSL https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.50.0-stable.tar.xz | tar xJ -C /sdk
  git config --global --add safe.directory "*"; export PATH=/sdk/flutter/bin:$PATH
  flutter test && flutter build apk --debug'
```

The APK is written to `build/app/outputs/flutter-apk/app-debug.apk`.
`pubspec.lock` is the lock file; always commit it together with `pubspec.yaml`.
