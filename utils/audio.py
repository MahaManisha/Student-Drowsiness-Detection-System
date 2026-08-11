"""
Audio Alert Utility for Streamlit Interface.
Injects Web Audio API sound synthesizers into Streamlit for alert notifications.
"""
import streamlit as st
import streamlit.components.v1 as components


def trigger_audio_alert(alert_type: str = "drowsy"):
    """
    Inject JavaScript Web Audio API oscillator beep into Streamlit UI.

    Args:
        alert_type (str): 'drowsy' (loud red siren) or 'distracted' (warning chime)
    """
    if alert_type == "drowsy":
        freq = 880  # A5 high pitch warning
        duration = 0.6
    else:
        freq = 587  # D5 warning tone
        duration = 0.3

    sound_script = f"""
    <script>
    (function() {{
        try {{
            var AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            var ctx = new AudioContext();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.value = {freq};
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + {duration});
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            osc.stop(ctx.currentTime + {duration});
        }} catch(e) {{
            console.log("Audio alert playback error:", e);
        }}
    }})();
    </script>
    """
    try:
        components.html(sound_script, height=0, width=0)
    except Exception:
        pass
