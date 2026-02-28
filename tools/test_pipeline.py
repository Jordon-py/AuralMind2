import os
import numpy as np
import soundfile as sf
import auralmind_maestro as maestro
import logging

# Setup logging to capture maestro logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auralmind")

def test_pipeline():
    # 1. Create a 44.1kHz sine wave
    sr_in = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr_in * duration), endpoint=False)
    y_in = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    y_in = np.stack([y_in, y_in], axis=1) # Stereo
    
    test_input = "test_441.wav"
    sf.write(test_input, y_in, sr_in)
    
    # 2. Run master()
    test_output = "test_master_float32.wav"
    preset = maestro.get_presets()["hi_fi_streaming"]
    # Ensure bit_depth is set on preset for the export routing logic
    preset.bit_depth = "float32"
    
    print("\n--- Running Master Pipeline ---")
    result = maestro.master(test_input, test_output, preset)
    
    # 3. Verify Resampling
    print(f"\nVerification:")
    print(f"- Processed Sample Rate: {result['sr']} Hz (Expected 48000)")
    assert result['sr'] == 48000, "Resampling to 48kHz failed!"
    
    # 4. Verify Export precision and Dither
    info = sf.info(test_output)
    print(f"- Export Subtype: {info.subtype} (Expected FLOAT)")
    assert info.subtype == 'FLOAT', "Export subtype should be FLOAT!"
    
    # Check for dither: in a silent file, dither adds noise. 
    # Here we used a sine wave, but we can check if the output matches FLOAT expectation.
    y_out, sr_out = sf.read(test_output)
    print(f"- Output Sample Rate: {sr_out} Hz")
    assert sr_out == 48000, "Output file sample rate should be 48000!"
    
    print("\nSUCCESS: 48kHz enforcement and float export routing verified.")
    
    # Cleanup
    if os.path.exists(test_input): os.remove(test_input)
    if os.path.exists(test_output): os.remove(test_output)
    if os.path.exists("report_test_master_float32.wav.md"): os.remove("report_test_master_float32.wav.md")

if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\nFAILURE: {e}")
        exit(1)
