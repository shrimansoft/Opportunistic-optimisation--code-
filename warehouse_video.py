#!/usr/bin/env python3
"""
Warehouse Simulation Video Creator

This script runs your warehouse simulation and automatically generates a video
without needing external FFmpeg commands. It integrates with your existing
warehouse simulation code.
"""

import os
import argparse
from pathlib import Path
from video_generator import VideoGenerator


def run_simulation_with_video(
    steps=100,
    output_video="warehouse_simulation.mp4",
    frames_dir="simulation_frames",
    framerate=10,
    cleanup_frames=True,
    save_every=1,
):
    """
    Run warehouse simulation and create video.

    Args:
        steps: Number of simulation steps
        output_video: Output video filename
        frames_dir: Directory to store frames
        framerate: Video framerate (fps)
        cleanup_frames: Whether to delete frames after video creation
        save_every: Save frame every N steps (1 = every step)
    """
    print(f"Running warehouse simulation for {steps} steps...")

    # Import your simulation modules
    try:
        from warehouse_sim.warehouse import Warehouse
        from warehouse_sim.environment import WarehouseEnv
    except ImportError as e:
        print(f"Error importing simulation modules: {e}")
        print("Make sure you're running from the project root directory")
        return False

    # Create frames directory
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize simulation
    warehouse = Warehouse()
    env = WarehouseEnv(warehouse)

    # Reset environment
    obs, info = env.reset()

    frame_count = 0
    print("Starting simulation...")

    for step in range(steps):
        # Take a random action (you can replace this with your policy)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        # Save frame every N steps
        if step % save_every == 0:
            # Use the warehouse's existing plotting method
            warehouse.shelf_plot(frames_dir)
            frame_count += 1

        # Progress indicator
        if step % 10 == 0:
            print(f"Step {step}/{steps} - Frames: {frame_count}")

        # Check if simulation ended
        if terminated or truncated:
            print(f"Simulation ended early at step {step}")
            break

    print(f"Simulation complete! Generated {frame_count} frames")

    # Create video
    print("Creating video...")
    generator = VideoGenerator()

    success = generator.create_video_from_frames(
        frames_dir=frames_dir,
        output_path=output_video,
        framerate=framerate,
        frame_pattern="frame_%04d.png",
        quality="high",
    )

    if success:
        print(f"✓ Video created: {output_video}")

        # Show file info
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
            print(f"  Duration: ~{frame_count/framerate:.1f} seconds")
    else:
        print("✗ Failed to create video")
        return False

    # Cleanup frames if requested
    if cleanup_frames and success:
        try:
            import shutil

            shutil.rmtree(frames_dir)
            print(f"Cleaned up frames from '{frames_dir}'")
        except Exception as e:
            print(f"Warning: Could not clean up frames: {e}")

    return success


def convert_existing_frames(
    frames_dir,
    output_video="converted_video.mp4",
    framerate=10,
    frame_pattern="frame_%04d.png",
):
    """
    Convert existing frames to video.

    Args:
        frames_dir: Directory containing existing frames
        output_video: Output video filename
        framerate: Video framerate
        frame_pattern: Pattern for frame filenames
    """
    if not os.path.exists(frames_dir):
        print(f"Error: Frames directory '{frames_dir}' does not exist")
        return False

    print(f"Converting frames from '{frames_dir}' to video...")

    generator = VideoGenerator()
    success = generator.create_video_from_frames(
        frames_dir=frames_dir,
        output_path=output_video,
        framerate=framerate,
        frame_pattern=frame_pattern,
        quality="high",
    )

    if success:
        print(f"✓ Video created: {output_video}")
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
    else:
        print("✗ Failed to create video")

    return success


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Create videos from warehouse simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation and create video
  python warehouse_video.py simulate --steps 200 --output my_video.mp4
  
  # Convert existing frames to video
  python warehouse_video.py convert shelfs2 --output warehouse_shelf_distribution4.mp4
  
  # Quick demo with 50 steps
  python warehouse_video.py simulate --steps 50 --framerate 15
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Simulate command
    sim_parser = subparsers.add_parser(
        "simulate", help="Run simulation and create video"
    )
    sim_parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of simulation steps (default: 100)",
    )
    sim_parser.add_argument(
        "--output", default="warehouse_simulation.mp4", help="Output video file"
    )
    sim_parser.add_argument(
        "--framerate", type=int, default=10, help="Video framerate (default: 10)"
    )
    sim_parser.add_argument(
        "--frames-dir", default="simulation_frames", help="Temporary frames directory"
    )
    sim_parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep frame files after video creation",
    )
    sim_parser.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="Save frame every N steps (default: 1)",
    )

    # Convert command
    conv_parser = subparsers.add_parser(
        "convert", help="Convert existing frames to video"
    )
    conv_parser.add_argument("frames_dir", help="Directory containing frame images")
    conv_parser.add_argument(
        "--output", default="converted_video.mp4", help="Output video file"
    )
    conv_parser.add_argument(
        "--framerate", type=int, default=10, help="Video framerate (default: 10)"
    )
    conv_parser.add_argument(
        "--pattern", default="frame_%04d.png", help="Frame filename pattern"
    )

    args = parser.parse_args()

    if args.command == "simulate":
        success = run_simulation_with_video(
            steps=args.steps,
            output_video=args.output,
            frames_dir=args.frames_dir,
            framerate=args.framerate,
            cleanup_frames=not args.keep_frames,
            save_every=args.save_every,
        )
        exit(0 if success else 1)

    elif args.command == "convert":
        success = convert_existing_frames(
            frames_dir=args.frames_dir,
            output_video=args.output,
            framerate=args.framerate,
            frame_pattern=args.pattern,
        )
        exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
