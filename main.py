import cv2
import os

# ==========================================================
# 🔹 PIPELINE FILTERS
# ==========================================================

def append_video(frames, video_path):
    """Append frames from a video file to the current frame list."""
    full_path = os.path.join("input", video_path)
    cap = cv2.VideoCapture(full_path)
    if not cap.isOpened():
        print(f"❌ Error: cannot open {full_path}")
        return frames

    new_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        new_frames.append(frame)
    cap.release()

    print(f"📹 Appended {len(new_frames)} frames from {video_path}")
    return frames + new_frames


def append_image(frames, image_path):
    """Append a static image as one frame."""
    full_path = os.path.join("input", image_path)
    img = cv2.imread(full_path)
    if img is None:
        print(f"❌ Error: cannot open image {image_path}")
        return frames
    print(f"🖼️ Appended single image {image_path}")
    return frames + [img]


def cut(frames, start, count):
    """
    Remove a range of frames relative to the last frame at the moment of the cut.
    - start: how many frames back from the last frame to start removing
    - count: number of frames to remove backward from start
    """
    assert start >= 0 and count >= 0, "cut() indices must be non-negative"
    end_idx = len(frames) - start
    start_idx = max(0, end_idx - count)
    new_frames = frames[:start_idx] + frames[end_idx:]
    print(f"✂️ Cut {count} frames starting {start} frames back "
          f"(removed frames {start_idx}:{end_idx})")
    return new_frames


def to_grayscale(frames):
    return [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]


def ensure_3channel(frames):
    """Convert grayscale frames back to 3-channel for display/saving."""
    return [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if len(f.shape) == 2 else f for f in frames]


def blur_frames(frames, ksize=(7, 7)):
    return [cv2.GaussianBlur(f, ksize, 0) for f in frames]


def overlay_text(frames, text, position=(50, 50), color=(0, 255, 0)):
    """Draw text label on each frame."""
    for f in frames:
        cv2.putText(f, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frames


# ==========================================================
# 🔹 PIPELINE ENGINE
# ==========================================================

def apply_pipeline(frames, pipeline):
    """Apply a sequence of transformation functions to a list of frames."""
    for step in pipeline:
        func = step[0]
        args = step[1:] if len(step) > 1 else []
        print(f"⚙️ Applying: {func.__name__}{tuple(args)}")
        frames = func(frames, *args)
    return frames


# ==========================================================
# 🔹 PLAYER + SAVING SYSTEM
# ==========================================================

def save_video(frames, output_path="output.mp4", fps=30):
    """Save processed frames as an MP4 video."""
    if not frames:
        print("⚠️ No frames to save.")
        return

    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
    )
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"✅ Saved {len(frames)} frames to {output_path}")


def play_video(frames, delay=30):
    """Play frames with basic controls."""
    if not frames:
        print("⚠️ No frames to display.")
        return

    current = 0
    playing = True

    while True:
        cv2.imshow("Video Player", frames[current])
        key = cv2.waitKey(delay) & 0xFF

        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE - toggle pause
            playing = not playing
        elif key == 81:  # left arrow
            current = max(0, current - 1)
        elif key == 83:  # right arrow
            current = min(len(frames) - 1, current + 1)

        if playing:
            current = (current + 1) % len(frames)

    cv2.destroyAllWindows()


# ==========================================================
# 🔹 MAIN
# ==========================================================

def main():
    # Define your single pipeline here
    pipeline = [
        (append_video, "maw.webm"),
        (cut, 300, 50),
        (cut, 100, 100),
        (append_video, "food.webm"),
        (cut, 100, 200),
        (to_grayscale,), # TODO: Add start and count arguments
        (ensure_3channel,), # TODO: Add start and count arguments
        (append_image, "huh.png"), # TODO: Add count argument
        (append_image, "drool.gif"),
        (blur_frames, (9, 9)), # TODO: Add start and count arguments
        (overlay_text, "Cats", (30, 60), (255, 200, 200)), # TODO: Add start and count arguments
    ]

    print("Controls:")
    print("  SPACE - Play/Pause")
    print("  ←/→   - Step backward/forward")
    print("  ESC   - Exit\n")

    frames = []
    processed_frames = apply_pipeline(frames, pipeline)

    save_video(processed_frames, "output.mp4")
    play_video(processed_frames)


if __name__ == "__main__":
    main()
