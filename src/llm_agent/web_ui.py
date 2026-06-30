#!/usr/bin/env python3
"""
Web UI for LLM-Controlled Battery Disassembly Robot
Version 2: Simplified async handling for Gradio compatibility
"""
import gradio as gr
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from executor import Executor
from planner import Planner
from validator import Validator
import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Global executor for running async tasks
executor_pool = ThreadPoolExecutor(max_workers=4)

def run_async(coro):
    """Run async coroutine in a separate thread with its own event loop"""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    future = executor_pool.submit(run_in_thread)
    return future.result()

class RobotUI:
    """
    Web-based user interface for controlling the battery disassembly robot.

    This class provides a Gradio interface for users to interact with the robot
    using natural language commands. It coordinates between the LLM planner,
    validator, and executor components.

    Attributes:
        executor (Executor): Handles ROS2 communication and skill execution
        planner (Planner): Converts natural language to robot skill sequences
        validator (Validator): Validates planned skills against safety rules
        system_running (bool): Tracks whether ROS2 connection is initialized

    Methods:
        initialize_system(): Establishes ROS2 connection and initializes components
        execute_prompt(user_prompt): Processes user commands end-to-end
        shutdown_system(): Cleanly shuts down ROS2 connections
    """
    def __init__(self):
        self.executor = None
        self.planner = None
        self.validator = None
        self.system_running = False

    def initialize_system(self):
        """Initialize ROS2 connection"""
        if not self.system_running:
            try:
                print("🔧 Initializing system...")
                self.executor = Executor(use_ros=True)
                print("   ✅ Executor initialized")

                self.planner = Planner(backend="openrouter", enable_rag=True)
                print("   ✅ Planner initialized")

                self.validator = Validator()
                print("   ✅ Validator initialized")

                self.system_running = True
                return "✅ System initialized successfully!"
            except Exception as e:
                import traceback
                error_msg = f"❌ Initialization failed: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                return error_msg
        return "⚠️  System already running"

    def execute_prompt(self, user_prompt, progress=gr.Progress()):
        """Execute user prompt"""
        print(f"\n{'='*60}")
        print(f"WEB UI: Received command: {user_prompt}")
        print(f"{'='*60}\n")

        if not self.system_running:
            result = "❌ Please initialize system first!", "", ""
            print(f"Returning: {result[0]}")
            return result

        if not user_prompt.strip():
            result = "⚠️  Please enter a command", "", ""
            print(f"Returning: {result[0]}")
            return result

        log = []
        log.append(f"📝 Your Command: {user_prompt}\n")

        start_time = time.time()

        # Step 1: Planning (with RAG)
        progress(0.2, desc="Planning...")
        log.append("🧠 Planning phase started...")

        if self.planner.rag and self.planner.rag.enabled:
            log.append("  ├─ Retrieving similar cases from knowledge base...")

        log.append("  ├─ Sending request to LLM...")

        try:
            # Run async planning in thread pool
            print("Calling planner.plan()...")
            plan = run_async(self.planner.plan(user_prompt))
            print(f"Plan received: {plan.get('plan', [])}")

            num_steps = len(plan.get('plan', []))

            # Display RAG context
            rag_context = plan.get('rag_context', [])
            if rag_context and len(rag_context) > 0:
                log.append(f"  ├─ Found {len(rag_context)} similar case(s) in knowledge base:")
                for i, case in enumerate(rag_context, 1):
                    score = case['similarity_score']
                    task_preview = case['task'][:40] + "..." if len(case['task']) > 40 else case['task']
                    log.append(f"  │  [{i}] Similarity: {score:.2f} | \"{task_preview}\"")
            elif self.planner.rag and self.planner.rag.enabled:
                log.append("  ├─ No similar cases found in knowledge base")

            log.append(f"  ├─ LLM responded successfully")
            log.append(f"  └─ Generated {num_steps}-step plan")
            log.append(f"✅ Planning completed\n")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            log.append(f"  └─ ❌ LLM request failed")
            log.append(f"❌ Planning failed: {str(e)}")
            print(f"Planning error:\n{error_trace}")
            result = "\n".join(log), "", json.dumps({"error": str(e)}, indent=2)
            print(f"Returning error result")
            return result

        # Step 2: Validation
        progress(0.4, desc="Validating...")
        log.append("🔍 Validating plan...")
        log.append("  ├─ Checking plan structure...")
        log.append("  ├─ Validating skills and parameters...")

        for i, action in enumerate(plan.get('plan', []), 1):
            skill = action.get('name', 'unknown')
            params = action.get('params', {})
            target = params.get('target', 'N/A')
            log.append(f"  │  Step {i}: {skill}(target={target})")

        log.append("  ├─ Checking safety constraints...")
        is_valid, errors = self.validator.validate_plan(plan)

        if not is_valid:
            log.append("  └─ ❌ Validation FAILED\n")
            log.append("Validation Errors:")
            for error in errors:
                log.append(f"  • {error}")
            log.append("")
            result = "\n".join(log), "", json.dumps(plan, indent=2)
            print(f"Validation failed, returning")
            return result

        log.append("  └─ ✅ All checks passed!")
        log.append("✅ Plan validation successful\n")

        # Step 3: Execution
        progress(0.6, desc="Executing...")
        log.append("🚀 Execution phase started...")
        log.append("  ├─ Sending commands to robot...")

        plan_text = self._format_plan(plan)

        try:
            print("Executing plan...")
            results = self.executor.execute(plan, timeout=20.0)
            print(f"Execution results: {results}")

            # Format results
            progress(1.0, desc="Complete!")
            log.append("  └─ Robot execution completed\n")
            log.append("📊 Execution Results:")
            total_steps = results['executed'] + results['failed']
            log.append(f"  ├─ Total steps: {total_steps}")
            log.append(f"  ├─ Successful: {results['executed']}")
            log.append(f"  ├─ Failed: {results['failed']}")
            if total_steps > 0:
                success_rate = results['executed'] / total_steps * 100
                log.append(f"  └─ Success rate: {success_rate:.0f}%")

            if results['success']:
                log.append("\n🎉 Task completed successfully!")

                # Save to RAG
                if self.planner.rag and self.planner.rag.enabled:
                    execution_time = time.time() - start_time
                    try:
                        self.planner.rag.add_successful_case(
                            task=user_prompt,
                            plan=plan,
                            execution_time=execution_time
                        )
                        log.append(f"💾 Saved to knowledge base (exec time: {execution_time:.1f}s)")
                    except Exception as e:
                        log.append(f"⚠️  Failed to save to knowledge base: {e}")
            else:
                log.append("\n❌ Task execution failed")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            log.append(f"  └─ ❌ Robot communication error")
            log.append(f"\n❌ Execution error: {str(e)}")
            print(f"Execution error:\n{error_trace}")
            result = "\n".join(log), plan_text, json.dumps(plan, indent=2)
            print(f"Returning execution error")
            return result

        final_result = ("\n".join(log), plan_text, json.dumps(results, indent=2))
        print(f"Returning success result (log length: {len(final_result[0])})")
        return final_result

    def _format_plan(self, plan):
        """Format plan for display"""
        lines = ["Step-by-step Plan:", "=" * 50]
        for step in plan.get('plan', []):
            skill = step['name']
            target = step['params'].get('target', '')
            target_str = f" → {target}" if target else ""
            lines.append(f"{step['step']}. {skill}{target_str}")
        return "\n".join(lines)

    def get_robot_state(self):
        """Get current robot state"""
        if not self.system_running or not self.executor:
            return "System not initialized"

        state = self.executor.get_current_state()
        if not state:
            return "No state available"

        lines = [
            "🤖 Current Robot State",
            "=" * 50,
            f"Gripper: {state['gripper_state']}",
            "",
            "Arm Joints:"
        ]
        for joint, pos in state['arm_joints'].items():
            lines.append(f"  {joint}: {pos:.4f} rad")

        return "\n".join(lines)

# Create UI instance
robot_ui = RobotUI()

# Build Gradio Interface
with gr.Blocks(title="LLM Robot Control") as demo:
    gr.Markdown("""
    # 🤖 LLM-Controlled Battery Disassembly Robot (v2)

    Control the robot using natural language commands powered by AI.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Control Panel
            gr.Markdown("## 🎮 Control Panel")

            init_btn = gr.Button("🔌 Initialize System", variant="primary", size="lg")
            init_output = gr.Textbox(label="System Status", lines=2)

            gr.Markdown("---")

            prompt_input = gr.Textbox(
                label="💬 Command",
                placeholder="Example: Remove the bolt and place it in the tray",
                lines=2
            )

            execute_btn = gr.Button("▶️ Execute", variant="primary", size="lg")

            gr.Markdown("### Quick Commands:")
            with gr.Row():
                gr.Button("🏠 Go Home", size="sm").click(
                    lambda: "Go to home position",
                    outputs=prompt_input
                )
                gr.Button("🔩 Remove Bolt", size="sm").click(
                    lambda: "Remove the bolt and place it in the tray",
                    outputs=prompt_input
                )
                gr.Button("👁️ Observe", size="sm").click(
                    lambda: "Move to observation position",
                    outputs=prompt_input
                )

        with gr.Column(scale=1):
            # Status Panel
            gr.Markdown("## 📊 Robot Status")
            state_output = gr.Textbox(label="Current State", lines=12)
            refresh_btn = gr.Button("🔄 Refresh State")

    # Output Panel
    gr.Markdown("---")
    gr.Markdown("## 📋 Execution Log")

    with gr.Row():
        log_output = gr.Textbox(label="Console Output", lines=15)
        plan_output = gr.Textbox(label="Generated Plan", lines=15)

    with gr.Accordion("📄 Detailed Results (JSON)", open=False):
        json_output = gr.Code(label="Raw Data", language="json")

    # Examples
    gr.Examples(
        examples=[
            ["Remove the bolt and place it in the tray"],
            ["Go to home position"],
            ["Move to observation position"],
            ["Pick up the bolt"],
            ["Place object in tray"],
        ],
        inputs=prompt_input,
        label="Example Commands"
    )

    # Event handlers
    init_btn.click(
        fn=robot_ui.initialize_system,
        outputs=init_output
    )

    execute_btn.click(
        fn=robot_ui.execute_prompt,
        inputs=prompt_input,
        outputs=[log_output, plan_output, json_output]
    )

    refresh_btn.click(
        fn=robot_ui.get_robot_state,
        outputs=state_output
    )

    gr.Markdown("""
    ---
    ### 📖 Instructions:
    1. Click **Initialize System** first
    2. Enter your command in natural language
    3. Click **Execute** to run
    4. Watch the robot execute in RViz window

    ### ⚙️ Available Actions:
    - **Positions**: HOME, approach_bolts, place_bolts
    - **Skills**: moveTo, grasp, release, openGripper, closeGripper
    """)

# Launch
if __name__ == "__main__":
    print("🚀 Starting Web UI (V2 - Simplified Async)...")
    print("📍 Open browser at: http://localhost:7862")
    print("🤖 Make sure ROS2 system is running!")
    print()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        theme=gr.themes.Soft()
    )
