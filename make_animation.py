import os
import re
import cv2
import numpy as np
import argparse

def center_crop_and_resize(img, crop_size, out_size):
    h, w = img.shape[:2]
    center_h, center_w = h // 2, w // 2
    half = crop_size // 2
    top = max(center_h - half, 0)
    left = max(center_w - half, 0)
    cropped = img[top:top+crop_size, left:left+crop_size]
    resized = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return resized

def blend_overlay(background, overlay, alpha):
    comp = background.copy().astype(np.float32)
    h_bg, w_bg = comp.shape[:2]
    h_ov, w_ov = overlay.shape[:2]
    top = (h_bg - h_ov) // 2
    left = (w_bg - w_ov) // 2

    roi = comp[top:top+h_ov, left:left+w_ov]
    overlay_f = overlay.astype(np.float32)
    blended_roi = (1 - alpha) * roi + alpha * overlay_f
    comp[top:top+h_ov, left:left+w_ov] = blended_roi
    return comp.astype(np.uint8)

def create_continuous_zoom_video(folder, zoom_factor, fps, frames_per_transition, output_path):
    valid_exts = (".png", ".jpg", ".jpeg")
    items = []
    
    for fname in os.listdir(folder):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in valid_exts:
            continue
        m = re.match(r'^(\d+)_', name)
        if m:
            number = int(m.group(1))
            items.append((number, fname))
    if not items:
        print("No integer-leading images found in the folder!")
        return
    
    items.sort(key=lambda x: x[0])
    
    images = []
    for _, fname in items:
        path = os.path.join(folder, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not read {path}, skipping.")
            continue
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        images.append(img)
    
    if len(images) < 2:
        print("Need at least 2 images to create a transition video.")
        return
    
    num_transitions = len(images) - 1
    total_frames = num_transitions * frames_per_transition
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (1024, 1024))
    
    for f in range(total_frames):
        T = f / float(frames_per_transition)  # T in [0, num_transitions)
        i = int(T)
        t = T - i  # local time in [0, 1]
        
        s = 1.0 + (zoom_factor - 1.0) * t  # background zoom scale
        
        crop_size = int(round(1024 / s))
        background = center_crop_and_resize(images[i], crop_size, 1024)
        overlay_size = int(round((1024 * s) / zoom_factor))
        overlay = cv2.resize(images[i+1], (overlay_size, overlay_size), interpolation=cv2.INTER_LINEAR)
        frame_img = blend_overlay(background, overlay, alpha=t)
        out.write(frame_img)
    
    hold_frames = int(0.5 * fps)
    for _ in range(hold_frames):
        out.write(images[-1])
    
    out.release()
    print(f"Video saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Create a continuous synchronized zoom video from integer-leading images.")
    parser.add_argument("folder", help="Folder containing the images (must be 1024x1024 and start with an integer prefix).")
    parser.add_argument("--zoom_factor", type=float, default=2.0,
                        help="Zoom factor p (default: 2.0).")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30).")
    parser.add_argument("--frames_per_transition", type=int, default=60,
                        help="Number of frames for each transition (default: 60).")
    parser.add_argument("--output", default=None,
                        help="Output video file path. By default, saved as 'animation.mp4' in the image folder.")
    
    args = parser.parse_args()
    
    output_path = args.output if args.output else os.path.join(args.folder, "animation.mp4")
    
    create_continuous_zoom_video(args.folder, args.zoom_factor, args.fps, args.frames_per_transition, output_path)

if __name__ == "__main__":
    main()
