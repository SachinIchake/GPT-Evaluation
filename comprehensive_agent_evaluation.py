"""
DESCRIPTION:
    Comprehensive end-to-end agent evaluation script using Azure AI Projects SDK v2.
    This script:
    1. Executes agents with function tools end-to-end
    2. Captures real agent responses and tool calls
    3. Creates a dataset from the captured data
    4. Runs multiple agentic evaluators using the dataset ID
    
    Evaluators run:
    - Tool-related: tool_call_accuracy, tool_selection, tool_input_accuracy, 
                    tool_output_utilization, tool_call_success
    - Task-related: task_completion, task_adherence
    - Quality-related: coherence, fluency, relevance, groundedness, intent_resolution

USAGE:
    python comprehensive_agent_evaluation.py

    Before running:
    pip install "azure-ai-projects>=2.0.0b1" python-dotenv

    Set these environment variables:
    1) AZURE_AI_PROJECT_ENDPOINT - Required
    2) AZURE_AI_MODEL_DEPLOYMENT_NAME - Required (for evaluation)
    3) DATASET_NAME - Optional (default: auto-generated)
    4) DATASET_VERSION - Optional (default: "1")
"""

import os
import json
import time
from pprint import pprint
from datetime import datetime
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import DatasetVersion
from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileID,
)
from openai.types.eval_create_params import DataSourceConfigCustom

load_dotenv()
 
# ========================================
# Evaluator Configurations
# ========================================

# Schema mappings for each evaluator
EVALUATOR_SCHEMAS = {
    "tool_call_accuracy": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_calls": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "tool_definitions"],
    },
    "tool_selection": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_calls": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response", "tool_definitions"],
    },
    "tool_input_accuracy": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response", "tool_definitions"],
    },
    "tool_output_utilization": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response"],
    },
    "tool_call_success": {
        "properties": {
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["response"],
    },
    "task_completion": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response"],
    },
    "task_adherence": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response"],
    },
    "coherence": {
        "properties": {
            "query": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["query", "response"],
    },
    "fluency": {
        "properties": {
            "query": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["response"],
    },
    "relevance": {
        "properties": {
            "query": {"type": "string"},
            "response": {"type": "string"}
        },
        "required": ["query", "response"],
    },
    "groundedness": {
        "properties": {
            "context": {"type": "string"},
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["response"],
    },
    "intent_resolution": {
        "properties": {
            "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
            "tool_definitions": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        },
        "required": ["query", "response"],
    },
}

# Data mappings for each evaluator
EVALUATOR_DATA_MAPPINGS = {
    "tool_call_accuracy": {
        "query": "{{item.query}}",
        "tool_definitions": "{{item.tool_definitions}}",
        "tool_calls": "{{item.tool_calls}}",
        "response": "{{item.response}}",
    },
    "tool_selection": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_calls": "{{item.tool_calls}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "tool_input_accuracy": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "tool_output_utilization": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "tool_call_success": {
        "tool_definitions": "{{item.tool_definitions}}",
        "response": "{{item.response}}",
    },
    "task_completion": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "task_adherence": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "coherence": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
    },
    "fluency": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
    },
    "relevance": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
    },
    "groundedness": {
        "context": "{{item.context}}",
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "intent_resolution": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
    "tool_output_utilization": {
        "query": "{{item.query}}",
        "response": "{{item.response}}",
        "tool_definitions": "{{item.tool_definitions}}",
    },
}


def get_unified_data_source_config() -> DataSourceConfigCustom:
    """Get a unified data source config that supports all evaluators."""
    # Combine all unique properties from all evaluators
    all_properties = {
        "query": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
        "response": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "object"}}]},
        "context": {"type": "string"},
        "tool_definitions": {"anyOf": [{"type": "string"}, {"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        "tool_calls": {"anyOf": [{"type": "object"}, {"type": "array", "items": {"type": "object"}}]},
        "ground_truth": {"type": "string"},
    }
    
    return DataSourceConfigCustom({
        "type": "custom",
        "item_schema": {
            "type": "object",
            "properties": all_properties,
            "required": [],
        },
        "include_sample_schema": True,
    })


def build_testing_criteria(evaluator_names: list[str], model_deployment_name: str) -> list[dict]:
    """Build testing criteria for multiple evaluators."""
    testing_criteria = []
    
    for evaluator_name in evaluator_names:
        if evaluator_name not in EVALUATOR_DATA_MAPPINGS:
            print(f"Warning: Unknown evaluator '{evaluator_name}', skipping...")
            continue
        
        testing_criteria.append({
            "type": "azure_ai_evaluator",
            "name": evaluator_name,
            "evaluator_name": f"builtin.{evaluator_name}",
            "initialization_parameters": {"deployment_name": model_deployment_name},
            "data_mapping": EVALUATOR_DATA_MAPPINGS[evaluator_name],
        })
    
    return testing_criteria


# get_evaluator_configs removed (unused)


def main():
    endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    model_deployment_name = os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-4.1")
    dataset_name = os.environ.get("DATASET_NAME", f"agent-eval-{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")
    dataset_version = os.environ.get("DATASET_VERSION", "2")
    
    with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=endpoint, credential=credential) as project_client,
        project_client.get_openai_client() as client,
    ):
               
        print("\n" + "="*80)
        print("STEP 1: Create Dataset from Captured Data")
        print("="*80)
        test_cases = []
        
        # # Create JSONL file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            with open('D:/MS/Work/Bajaj/Voice Bot/Evaluation/generated_test_dataset_V1.jsonl', 'r') as sample_file:
                for line in sample_file:
                    test_case = json.loads(line)
                    test_cases.append(test_case)
            for test_case in test_cases:
                f.write(json.dumps(test_case) + '\n')
            temp_file_path = f.name
        
        
        # Upload dataset
        dataset: DatasetVersion = project_client.datasets.upload_file(
            name=dataset_name,
            version=dataset_version,
            file_path=temp_file_path,
        )
        print(f"Dataset created: {dataset.name} (version: {dataset.version}, id: {dataset.id})")
        
        # Clean up temp file
        # os.unlink(temp_file_path)
        
        print("\n" + "="*80)
        print("STEP 3: Create and Run Combined Evaluation")
        print("="*80)
        
        # Select evaluators to run (matching the screenshot)
        evaluators_to_run = [
            "tool_call_accuracy",
            "tool_selection",
            "tool_input_accuracy",
            "tool_call_success",
            "task_completion",
            "task_adherence",
            "coherence",
            "fluency",
            "relevance",
            "groundedness",
            "intent_resolution",
            "tool_output_utilization",
        ]
        
        # Build testing criteria with all evaluators
        print(f"Building testing criteria for {len(evaluators_to_run)} evaluators...")
        testing_criteria = build_testing_criteria(evaluators_to_run, model_deployment_name)
        
        for criterion in testing_criteria:
            print(f"  ✓ {criterion['name']}")
        
        # Get unified data source config
        unified_data_source_config = get_unified_data_source_config()
        
        # Create single evaluation with all testing criteria
        print(f"\nCreating evaluation with {len(testing_criteria)} evaluators...")
        eval_object = client.evals.create(
            name=f"Comprehensive Agent Evaluation - {dataset_name}",
            data_source_config=unified_data_source_config,
            testing_criteria=testing_criteria,  # type: ignore
        )
        print(f"✓ Evaluation created (id: {eval_object.id})")
        
        # Create single evaluation run with dataset ID
        print(f"\nCreating evaluation run with dataset ID: {dataset.id}...")
        eval_run = client.evals.runs.create(
            eval_id=eval_object.id,
            name=f"comprehensive_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            metadata={
                "dataset": dataset_name,
                "evaluators": ", ".join(evaluators_to_run),
                "test_cases": len(test_cases),
            },
            data_source=CreateEvalJSONLRunDataSourceParam(
                type="jsonl",
                source=SourceFileID(type="file_id", id=dataset.id if dataset.id else "")
            ),
        )
        print(f"✓ Evaluation run created (id: {eval_run.id})")
        
        # Wait for completion
        print(f"\nWaiting for evaluation to complete...")
        while True:
            run = client.evals.runs.retrieve(run_id=eval_run.id, eval_id=eval_object.id)
            if run.status in ["completed", "failed"]:
                print(f"\n✓ Evaluation {run.status}")
                break
            print(".", end="", flush=True)
            time.sleep(5)
        
        # Get results
        evaluation_results = {
            "eval_id": eval_object.id,
            "run_id": eval_run.id,
            "status": run.status,
            "report_url": run.report_url,
            "result_counts": run.result_counts if hasattr(run, 'result_counts') else None,
        }
        
        print("\n\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Dataset: {dataset_name} (id: {dataset.id})")
        print(f"Test Cases: {len(test_cases)}")
        print(f"Evaluators Run: {len(evaluators_to_run)}")
        print(f"\nEvaluation Details:")
        print(f"  ID: {evaluation_results['eval_id']}")
        print(f"  Run ID: {evaluation_results['run_id']}")
        print(f"  Status: {evaluation_results['status']}")
        if evaluation_results['result_counts']:
            print(f"  Result Counts: {evaluation_results['result_counts']}")
        if evaluation_results['report_url']:
            print(f"\n  📊 Report URL: {evaluation_results['report_url']}")
        
        print(f"\nEvaluators included:")
        for evaluator in evaluators_to_run:
            print(f"  ✓ {evaluator}")
        
        # Get detailed output items
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print("="*80)
        output_items = list(client.evals.runs.output_items.list(
            run_id=evaluation_results['run_id'], 
            eval_id=evaluation_results['eval_id']
        ))
        # print(f"Total output items: {len(output_items)}")
        
        # if output_items:
        #     print(f"\nSample output (first item):")
        #     pprint(output_items[0])
        
        print("\n" + "="*80)
        print("COMPLETE")
        print("="*80)
        print(f"\n💡 View full results at: {evaluation_results['report_url']}")
        print(f"💡 Evaluation ID: {evaluation_results['eval_id']}")
        print(f"💡 Run ID: {evaluation_results['run_id']}")


if __name__ == "__main__":
    main()
