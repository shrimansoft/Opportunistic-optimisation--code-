#!/usr/bin/env python3
"""
Video Generator for Warehouse Simulation

This module provides functionality to generate video
It supports both OpenCV and MoviePy backends.
"""

import os
import glob
import argparse
from pathlib import Path
from typing import List

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from moviepy.editor import ImageSequenceClip

    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class VideoGenerator:
    """
    A class to generate videos from simulation frames.
    """

    def __init__(self, backend: str = "auto"):
        """
        Initialize the VideoGenerator.

        Args:
            backend (str): Video processing backend ('opencv', 'moviepy', or 'auto')
        """
        self.backend = self._select_backend(backend)

    def _select_backend(self, backend: str) -> str:
        """Select the appropriate backend based on availability."""
        if backend == "auto":
            if OPENCV_AVAILABLE:
                return "opencv"
            elif MOVIEPY_AVAILABLE:
                return "moviepy"
            else:
                raise ImportError(
                    "No video processing library available. "
                    "Please install opencv-python or moviepy."
                )
        elif backend == "opencv" and not OPENCV_AVAILABLE:
            raise ImportError(
                "OpenCV not available. Install with: pip install opencv-python"
            )
        elif backend == "moviepy" and not MOVIEPY_AVAILABLE:
            raise ImportError(
                "MoviePy not available. Install with: pip install moviepy"
            )

        return backend

    def create_video_from_frames(
        self,
        frames_dir: str,
        output_path: str,
        framerate: int = 10,
        frame_pattern: str = "frame_%04d.png",
        quality: str = "high",
    ) -> bool:
        """
        Create a video from frames in a directory.

        Args:
            frames_dir (str): Directory containing the frame images
            output_path (str): Output video file path
            framerate (int): Video framerate (frames per second)
            frame_pattern (str): Pattern for frame filenames
            quality (str): Video quality ('low', 'medium', 'high')

        Returns:
            bool: True if successful, False otherwise
        """
        frames_path = Path(frames_dir)
        if not frames_path.exists():
            print(f"Error: Frames directory '{frames_dir}' does not exist.")
            return False

        # Find all frame files
        frame_files = self._get_frame_files(frames_dir, frame_pattern)

        if not frame_files:
            print(
                f"Error: No frame files found in '{frames_dir}' with pattern '{frame_pattern}'"
            )
            return False

        print(f"Found {len(frame_files)} frames in '{frames_dir}'")
        print(f"Creating video using {self.backend} backend...")

        try:
            if self.backend == "opencv":
                return self._create_video_opencv(
                    frame_files, output_path, framerate, quality
                )
            elif self.backend == "moviepy":
                return self._create_video_moviepy(
                    frame_files, output_path, framerate, quality
                )
        except Exception as e:
            print(f"Error creating video: {e}")
            return False

        return False

    def _get_frame_files(self, frames_dir: str, pattern: str) -> List[str]:
        """Get sorted list of frame files."""
        # Convert pattern to glob pattern
        glob_pattern = pattern.replace("%04d", "*")
        frame_files = glob.glob(os.path.join(frames_dir, glob_pattern))

        # Sort numerically based on frame number
        def extract_frame_number(filename):
            try:
                # Extract number from filename
                basename = os.path.basename(filename)
                number_part = basename.replace("frame_", "").replace(".png", "")
                return int(number_part)
            except:
                return 0

        frame_files.sort(key=extract_frame_number)
        return frame_files

    def _create_video_opencv(
        self, frame_files: List[str], output_path: str, framerate: int, quality: str
    ) -> bool:
        """Create video using OpenCV."""
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Error: Could not read first frame '{frame_files[0]}'")
            return False

        height, width, _ = first_frame.shape

        # Set up video codec and quality
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Create VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))

        if not out.isOpened():
            print("Error: Could not open video writer")
            return False

        print(f"Processing {len(frame_files)} frames...")

        for i, frame_file in enumerate(frame_files):
            frame = cv2.imread(frame_file)
            if frame is None:
                print(f"Warning: Could not read frame '{frame_file}', skipping...")
                continue

            out.write(frame)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(frame_files)} frames")

        out.release()
        print(f"Video saved successfully to '{output_path}'")
        return True

    def _create_video_moviepy(
        self, frame_files: List[str], output_path: str, framerate: int, quality: str
    ) -> bool:
        """Create video using MoviePy."""
        print(f"Processing {len(frame_files)} frames...")

        # Create video clip from image sequence
        clip = ImageSequenceClip(frame_files, fps=framerate)

        # Set quality parameters
        if quality == "high":
            bitrate = "5000k"
        elif quality == "medium":
            bitrate = "2000k"
        else:  # low
            bitrate = "1000k"

        # Write video file
        clip.write_videofile(
            output_path,
            codec="libx264",
            audio=False,
            bitrate=bitrate,
            verbose=False,
            logger=None,
        )

        print(f"Video saved successfully to '{output_path}'")
        return True


class SimulationVideoGenerator(VideoGenerator):
    """
    Extended VideoGenerator specifically for warehouse simulation.
    """

    def run_simulation_and_create_video(
        self,
        steps: int = 100,
        output_video: str = "warehouse_simulation.mp4",
        frames_dir: str = "simulation_frames",
        framerate: int = 10,
        cleanup_frames: bool = True,
    ) -> bool:
        """
        Run a simulation, generate frames, and create a video.

        Args:
            steps (int): Number of simulation steps
            output_video (str): Output video filename
            frames_dir (str): Directory to store temporary frames
            framerate (int): Video framerate
            cleanup_frames (bool): Whether to delete frames after video creation

        Returns:
            bool: True if successful, False otherwise
        """
        from warehouse_sim.warehouse import Warehouse
        from warehouse_sim.environment import WarehouseEnv

        print(f"Running simulation for {steps} steps...")

        # Create frames directory
        os.makedirs(frames_dir, exist_ok=True)

        # Initialize warehouse and environment
        warehouse = Warehouse()
        env = WarehouseEnv(warehouse)

        # Reset and run simulation
        obs, info = env.reset()

        for step in range(steps):
            # Take random action (you can replace this with your policy)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Generate frame every step (or every few steps to reduce file size)
            if step % 1 == 0:  # Generate frame every step
                warehouse.enhanced_plot(frames_dir)

            if step % 10 == 0:
                print(f"Simulation step {step}/{steps}")

            if terminated or truncated:
                print(f"Simulation ended early at step {step}")
                break

        print(f"Simulation complete. Frames saved to '{frames_dir}'")

        # Create video from frames
        success = self.create_video_from_frames(
            frames_dir=frames_dir, output_path=output_video, framerate=framerate
        )

        if success and cleanup_frames:
            print(f"Cleaning up frames in '{frames_dir}'...")
            try:
                import shutil

                shutil.rmtree(frames_dir)
                print("Frames cleaned up successfully.")
            except Exception as e:
                print(f"Warning: Could not clean up frames: {e}")

        return success


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate videos from warehouse simulation"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subcommand for creating video from existing frames
    frames_parser = subparsers.add_parser(
        "from-frames", help="Create video from existing frames"
    )
    frames_parser.add_argument("frames_dir", help="Directory containing frame images")
    frames_parser.add_argument("output_video", help="Output video file path")
    frames_parser.add_argument(
        "--framerate", type=int, default=10, help="Video framerate (default: 10)"
    )
    frames_parser.add_argument(
        "--pattern", default="frame_%04d.png", help="Frame filename pattern"
    )
    frames_parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="high",
        help="Video quality",
    )
    frames_parser.add_argument(
        "--backend",
        choices=["opencv", "moviepy", "auto"],
        default="auto",
        help="Video backend",
    )

    # Subcommand for running simulation and creating video
    sim_parser = subparsers.add_parser(
        "simulate", help="Run simulation and create video"
    )
    sim_parser.add_argument(
        "--steps", type=int, default=100, help="Number of simulation steps"
    )
    sim_parser.add_argument(
        "--output", default="warehouse_simulation.mp4", help="Output video file"
    )
    sim_parser.add_argument(
        "--frames-dir", default="simulation_frames", help="Temporary frames directory"
    )
    sim_parser.add_argument("--framerate", type=int, default=10, help="Video framerate")
    sim_parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep frame files after video creation",
    )
    sim_parser.add_argument(
        "--backend",
        choices=["opencv", "moviepy", "auto"],
        default="auto",
        help="Video backend",
    )

    args = parser.parse_args()

    if args.command == "from-frames":
        generator = VideoGenerator(backend=args.backend)
        success = generator.create_video_from_frames(
            frames_dir=args.frames_dir,
            output_path=args.output_video,
            framerate=args.framerate,
            frame_pattern=args.pattern,
            quality=args.quality,
        )
        exit(0 if success else 1)

    elif args.command == "simulate":
        generator = SimulationVideoGenerator(backend=args.backend)
        success = generator.run_simulation_and_create_video(
            steps=args.steps,
            output_video=args.output,
            frames_dir=args.frames_dir,
            framerate=args.framerate,
            cleanup_frames=not args.keep_frames,
        )
        exit(0 if success else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
