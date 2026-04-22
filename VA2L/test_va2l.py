import argparse
from pathlib import Path
from PIL import Image

from VA2L.state_abstraction import prepare_state_abstraction_from_demo


def test_state_abstraction(args) -> None:
    """Test only state abstraction module: prepare and save VLM inputs."""
    print(f"\n=== Testing State Abstraction ===")
    print(f"Demo dir: {args.demo_dir}")
    print(f"Time step: {args.t}, History window: {args.k}")

    image, instruction, stats = prepare_state_abstraction_from_demo(
        demo_dir=args.demo_dir,
        t=args.t,
        window_size=args.k,
        manipulation_backend=args.backend,
        yolo_model_path=args.yolo_model_path,
    )

    out_image = Path(args.out_image)
    out_instruction = Path(args.out_instruction)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_instruction.parent.mkdir(parents=True, exist_ok=True)

    image.save(out_image)
    out_instruction.write_text(instruction, encoding="utf-8")

    print(f"✓ Saved image: {out_image}")
    print(f"✓ Saved instruction: {out_instruction}")
    print(f"✓ Detected objects: {', '.join(stats.get('detected_objects', [])) if stats.get('detected_objects') else 'none'}")
    print(f"\nInstruction:\n{instruction}")


def test_vlm_inference(args) -> None:
    """Test only VLM inference: load image/instruction and call the VLM."""
    from vlm_inference import VLMInference

    print(f"\n=== Testing VLM Inference ===")
    
    out_image = Path(args.out_image)
    out_instruction = Path(args.out_instruction)
    
    if not out_image.exists() or not out_instruction.exists():
        print(f"Error: State abstraction outputs not found. Run with --mode state_abstraction first.")
        return
    
    image = Image.open(out_image).convert("RGB")
    instruction = out_instruction.read_text(encoding="utf-8")
    
    print(f"Loaded image: {out_image}")
    print(f"Loaded instruction from: {out_instruction}")
    print(f"\nInitializing {args.model} model...")
    
    vlm = VLMInference(
        model=args.model,
        model_id=None,
        device=args.device,
        precision=args.precision,
    )
    result = vlm.infer(image, instruction)
    
    out_result = Path(str(out_instruction).replace(".txt", "_vlm_result.txt"))
    out_result.write_text(result, encoding="utf-8")
    
    print(f"\n✓ VLM Inference Result:")
    print(result)
    print(f"\n✓ Saved to: {out_result}")


def test_full_pipeline(args) -> None:
    """Test full pipeline: state abstraction + VLM inference."""
    print(f"\n=== Testing Full VA2L Pipeline ===")
    print(f"Demo dir: {args.demo_dir}")
    print(f"Time step: {args.t}, History window: {args.k}")
    
    print(f"\nStep 1: State Abstraction")
    image, instruction, stats = prepare_state_abstraction_from_demo(
        demo_dir=args.demo_dir,
        t=args.t,
        window_size=args.k,
        manipulation_backend=args.backend,
        yolo_model_path=args.yolo_model_path,
    )
    
    out_image = Path(args.out_image)
    out_instruction = Path(args.out_instruction)
    out_image.parent.mkdir(parents=True, exist_ok=True)
    out_instruction.parent.mkdir(parents=True, exist_ok=True)
    
    image.save(out_image)
    out_instruction.write_text(instruction, encoding="utf-8")
    print(f"✓ State abstraction done: {out_image}, {out_instruction}")
    print(f"✓ Detected objects: {', '.join(stats.get('detected_objects', [])) if stats.get('detected_objects') else 'none'}")
    
    print(f"\nStep 2: VLM Inference")
    from vlm_inference import VLMInference

    vlm = VLMInference(
        model=args.model,
        model_id=None,
        device=args.device,
        precision=args.precision,
    )
    result = vlm.infer(image, instruction)
    
    out_result = Path(str(out_instruction).replace(".txt", "_vlm_result.txt"))
    out_result.write_text(result, encoding="utf-8")
    
    print(f"✓ VLM inference done: {out_result}")
    print(f"\nInferred Task Intent:")
    print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="VA2L: Vision+Action → Language intent inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["state_abstraction", "vlm_inference", "full"],
        default="full",
        help="Test mode: state_abstraction (prepare VLM inputs), vlm_inference (call model), or full (end-to-end)",
    )
    parser.add_argument(
        "--demo_dir",
        type=str,
        default="dataset/robotiq_insert_tube",
        help="Path to demo folder with color/, trajectory.json, camera_intrinsics.json, camera_extrinsics.json",
    )
    parser.add_argument("--t", type=int, default=80, help="Time step to analyze")
    parser.add_argument("--k", type=int, default=20, help="History window size")
    parser.add_argument(
        "--out_image",
        type=str,
        default=None,
        help="Output visualization image path (default: <demo_dir>/debug/va2l_output_t<step>.png)",
    )
    parser.add_argument(
        "--out_instruction",
        type=str,
        default=None,
        help="Output instruction text path (default: <demo_dir>/debug/va2l_output_t<step>.txt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="VLM device selection: cuda:0 / cuda:1 / cpu",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen-vl-4b", "qwen-vl-8b", "qwen-vl-2b"],
        default="qwen-vl-4b",
        help="VLM model selection: qwen-vl-4b / qwen-vl-8b / qwen-vl-2b",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["auto", "fp16", "bf16", "fp32"],
        default="auto",
        help="Model precision: auto / fp16 / bf16 / fp32",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sam", "yolo"],
        default="sam",
        help="Manipulation backend for state abstraction: sam (current pipeline) or yolo.",
    )
    parser.add_argument(
        "--yolo_model_path",
        type=str,
        default="yolov8l-worldv2.pt",
        help="YOLO weights path used when --backend yolo is selected.",
    )
    args = parser.parse_args()

    if args.out_image is None:
        args.out_image = f"{args.demo_dir}/debug/va2l_output_t{args.t:04d}.png"
    if args.out_instruction is None:
        args.out_instruction = f"{args.demo_dir}/debug/va2l_output_t{args.t:04d}.txt"

    if args.mode == "state_abstraction":
        test_state_abstraction(args)
    elif args.mode == "vlm_inference":
        test_vlm_inference(args)
    elif args.mode == "full":
        test_full_pipeline(args)


if __name__ == "__main__":
    main()
