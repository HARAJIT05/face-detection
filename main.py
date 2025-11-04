#!/usr/bin/env python3
"""Main CLI entry point for face detection and recognition pipeline."""

import argparse
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

def main():
    """Main function to handle CLI arguments and dispatch to appropriate modules."""
    parser = argparse.ArgumentParser(
        description='Face Detection and Recognition Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Collect face data:
    python main.py --mode collect --name "John Doe" --samples 100
  
  Train the model:
    python main.py --mode train
  
  Run face recognition:
    python main.py --mode recognize
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['collect', 'train', 'recognize'],
        help='Operation mode: collect (gather face data), train (train model), or recognize (run recognition)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        help='Name of the person (required for collect mode)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=config.COLLECT_SAMPLES_COUNT,
        help=f'Number of face samples to collect (default: {config.COLLECT_SAMPLES_COUNT})'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=config.CAMERA_INDEX,
        help=f'Camera device index (default: {config.CAMERA_INDEX})'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        help='Video file path for recognition mode (default: use webcam)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Import modules only when needed
    if args.mode == 'collect':
        if not args.name:
            parser.error('--name is required for collect mode')
        
        from scripts.collect_data import FaceDataCollector
        collector = FaceDataCollector(camera_index=args.camera)
        collector.collect_samples(person_name=args.name, num_samples=args.samples)
    
    elif args.mode == 'train':
        from scripts.train_model import FaceModelTrainer
        trainer = FaceModelTrainer()
        trainer.train()
    
    elif args.mode == 'recognize':
        from scripts.face_pipeline import FaceRecognitionPipeline
        pipeline = FaceRecognitionPipeline(
            camera_index=args.camera if not args.source else None,
            video_source=args.source
        )
        pipeline.run()
    
    print("\n" + "="*50)
    print("Operation completed successfully!")
    print("="*50)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)
