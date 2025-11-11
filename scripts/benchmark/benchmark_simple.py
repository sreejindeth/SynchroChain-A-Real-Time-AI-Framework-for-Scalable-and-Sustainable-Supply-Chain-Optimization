"""
Simplified Async + Batched Benchmark
No user prompts, just runs and saves results
"""
import asyncio
import random
import time
import json
import os
from datetime import datetime
import sys
sys.path.insert(0, 'src/production')

from SynchroChain_Orchestrator_Batched import ModelOrchestrator, SupplyChainRequest

def make_request(i: int) -> SupplyChainRequest:
    """Generate a realistic supply chain request."""
    user_session = random.choices(
        ['view', 'search', 'add_to_cart', 'detail', 'compare'],
        k=random.randint(3, 8)
    )
    order_context = {
        'category_name': random.choice(['Electronics', 'Fashion', 'Home', 'Technology']),
        'shipping_mode': random.choice(['Standard Class', 'First Class', 'Same Day']),
        'order_quantity': random.randint(1, 5),
        'order_value': random.randint(50, 1500),
        'is_international': random.random() < 0.2
    }
    return SupplyChainRequest(
        session_id=f"SC_{i:06d}",
        user_session=user_session,
        order_context=order_context,
        timestamp=datetime.now(),
        priority="normal"
    )

async def run_benchmark(total_requests=5000, concurrency=256):
    """Run the benchmark."""
    print(f"\n{'='*70}")
    print(f"🚀 ASYNCHR ONOUS + BATCHED BENCHMARK")
    print(f"{'='*70}")
    print(f"Total Requests:     {total_requests}")
    print(f"Concurrency Level:  {concurrency}")
    print(f"Batch Size:         32")
    print(f"Batch Delay:        10ms")
    print(f"Target Throughput:  357 req/s")
    print(f"{'='*70}\n")
    
    # Initialize orchestrator
    print("⚙️  Initializing orchestrator with micro-batching...")
    orchestrator = ModelOrchestrator()
    orchestrator.intent_batcher.start()
    orchestrator.gnn_batcher.start()
    print("✅ Orchestrator initialized\n")
    
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    failures = 0
    
    async def process_one(idx):
        nonlocal failures
        async with sem:
            req = make_request(idx)
            t0 = time.perf_counter()
            resp = await orchestrator.process_supply_chain_request(req)
            dt = time.perf_counter() - t0
            latencies.append(dt * 1000)
            if not resp.success:
                failures += 1
    
    # Warmup
    print("🔥 Warmup (100 requests)...")
    warm_tasks = [asyncio.create_task(process_one(i)) for i in range(100)]
    await asyncio.gather(*warm_tasks)
    latencies.clear()
    failures = 0
    print("✅ Warmup complete\n")
    
    # Main benchmark
    print(f"📊 Running benchmark ({total_requests} requests)...")
    t_start = time.perf_counter()
    
    tasks = [asyncio.create_task(process_one(i)) for i in range(total_requests)]
    
    # Progress indicator
    completed = 0
    while completed < total_requests:
        await asyncio.sleep(0.5)
        completed = sum(1 for t in tasks if t.done())
        pct = (completed / total_requests) * 100
        print(f"\r   Progress: {completed}/{total_requests} ({pct:.1f}%)", end='', flush=True)
    
    await asyncio.gather(*tasks)
    print()  # New line after progress
    
    elapsed = time.perf_counter() - t_start
    
    # Calculate metrics
    latencies.sort()
    success_rate = ((total_requests - failures) / total_requests) * 100
    throughput = total_requests / elapsed
    
    p50 = latencies[int(0.50 * len(latencies)) - 1] if latencies else 0
    p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0
    p99 = latencies[int(0.99 * len(latencies)) - 1] if latencies else 0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    # Print results
    print(f"\n{'='*70}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"\n🎯 THROUGHPUT:")
    print(f"   Measured:  {throughput:.1f} req/s")
    print(f"   Target:    357.0 req/s")
    if throughput >= 357:
        print(f"   Status:    ✅ TARGET ACHIEVED ({(throughput/357)*100:.1f}% of target)")
    else:
        print(f"   Status:    ⚠️  {(throughput/357)*100:.1f}% of target")
    print(f"   vs Sequential: {throughput/61.1:.2f}x speedup (baseline was 61.1 req/s)")
    
    print(f"\n⏱️  LATENCY:")
    print(f"   Median (p50):  {p50:.1f} ms")
    print(f"   P95:           {p95:.1f} ms")
    print(f"   P99:           {p99:.1f} ms")
    print(f"   Average:       {avg_latency:.1f} ms")
    
    print(f"\n✅ RELIABILITY:")
    print(f"   Success Rate:  {success_rate:.2f}%")
    print(f"   Successful:    {total_requests - failures:,}")
    print(f"   Failed:        {failures}")
    
    print(f"\n⏳ TIMING:")
    print(f"   Total Time:    {elapsed:.2f} seconds")
    print(f"   Per Request:   {(elapsed/total_requests)*1000:.2f} ms")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    os.makedirs('results/system_performance', exist_ok=True)
    
    results = {
        "test_type": "Asynchronous + Micro-Batched",
        "timestamp": datetime.now().isoformat(),
        "throughput": {
            "measured_req_per_sec": round(throughput, 1),
            "target_req_per_sec": 357.0,
            "achieved_target": throughput >= 357,
            "percentage_of_target": round((throughput/357)*100, 1),
            "speedup_vs_sequential": round(throughput / 61.1, 2)
        },
        "latency": {
            "median_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(avg_latency, 2),
            "min_ms": round(min(latencies), 2) if latencies else 0,
            "max_ms": round(max(latencies), 2) if latencies else 0
        },
        "reliability": {
            "success_rate_percent": round(success_rate, 2),
            "total_requests": total_requests,
            "successful_requests": total_requests - failures,
            "failed_requests": failures,
            "error_rate_percent": round((failures/total_requests)*100, 2)
        },
        "timing": {
            "total_elapsed_sec": round(elapsed, 2),
            "avg_per_request_ms": round((elapsed/total_requests)*1000, 2)
        },
        "configuration": {
            "concurrency": concurrency,
            "batch_size": 32,
            "batch_delay_ms": 10,
            "total_requests": total_requests
        },
        "comparison": {
            "sequential_baseline": {
                "throughput_req_per_sec": 61.1,
                "median_latency_ms": 18.3
            },
            "batched_async": {
                "throughput_req_per_sec": round(throughput, 1),
                "median_latency_ms": round(p50, 2)
            },
            "improvement": {
                "throughput_speedup": round(throughput / 61.1, 2),
                "latency_change_ms": round(p50 - 18.3, 2)
            }
        }
    }
    
    result_file = 'results/system_performance/batched_benchmark_results.json'
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {result_file}\n")
    
    return results

async def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("  SynchroChain Async + Batched Performance Benchmark")
    print("  Objective: Validate 357 req/s throughput target")
    print("="*70)
    
    try:
        await run_benchmark(total_requests=5000, concurrency=256)
        print("🎉 Benchmark completed successfully!\n")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())





