# Chlorine Vision Mobile

This Expo app provides the phone interface for Chlorine Vision.

Core features:

- home screen
- capture or upload flow
- guided camera overlay
- on-device chlorine analysis
- result screen with target-range badge
- previous scan history with delete support

Current shipped mobile model:

- `Mobile Lasso Regression`
- exported from `training/train_mobile_deployable_model.py`
- loaded by `mobile/src/analyzer.ts`

## Run It

Start the Expo app:

```bash
cd mobile
npm install
npx expo start -c
```

Then open Expo Go on your phone or run the iOS simulator.

## Notes

- `Upload Existing Image` uses the phone photo library.
- `Take New Photo` uses the phone camera.
- Analysis now runs on-device using the exported mobile model in `mobile/src/mobile_model.json`.
- The app stores scan date, time, location, and concentration locally on the phone.
- The Python API is optional and is not required for normal app use.
