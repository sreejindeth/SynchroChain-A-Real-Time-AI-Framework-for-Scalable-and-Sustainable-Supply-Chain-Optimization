"""
High-Concurrency Benchmark for SynchroChain Orchestrator
Tests async + batched inference to validate target throughput (357 req/s)
"""
import asyncio
import random
import time
import json
from datetime import datetime
from statistics import mean, median, stdev
import sys
sys.path.insert(0, 'src/production')

from SynchroChain_Orchestrator_Batched import ModelOrchestrator, SupplyChainRequest

def make_request(i: int) -> SupplyChainRequest:
    """Generate a realistic supply chain request."""
    user_session = random.choices(
        ['view', 'search', 'add_to_cart', 'detail', 'compare', 'wishlist'],
        k=random.randint(3, 12)
    )
    order_context = {
        'category_name': random.choice(['Electronics', 'Fashion', 'Home', 'Technology', 'Sports']),
        'shipping_mode': random.choice(['Standard Class', 'First Class', 'Same Day']),
        'order_quantity': random.randint(1, 5),
        'order_value': random.randint(50, 1500),
        'is_international': random.random() < 0.2,
        'days_for_shipping': random.randint(1, 7),
        'days_for_shipment': random.randint(1, 5)
    }
    return SupplyChainRequest(
        session_id=f"SC_{i:06d}",
        user_session=user_session,
        order_context=order_context,
        timestamp=datetime.now(),
        priority="normal"
    )

async def run_load_test(orchestrator, total_requests=5000, concurrency=200):
    """Run high-concurrency load test."""
    print(f"\n{'='*70}")
    print(f"🚀 ASYNC + BATCHED LOAD TEST")
    print(f"{'='*70}")
    print(f"Total Requests: {total_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Target Throughput: 357 req/s")
    print(f"{'='*70}\n")
    
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    failures = 0
    
    async def one_call(idx):
        nonlocal failures
        async with sem:
            req = make_request(idx)
            t0 = time.perf_counter()
            resp = await orchestrator.process_supply_chain_request(req)
            dt = time.perf_counter() - t0
            latencies.append(dt * 1000)  # Convert to ms
            if not resp.success:
                failures += 1
    
    # Warmup phase
    print("⏳ Warmup phase (200 requests)...")
    warm_tasks = [asyncio.create_task(one_call(i)) for i in range(200)]
    await asyncio.gather(*warm_tasks)
    latencies.clear()
    failures = 0
    
    # Main benchmark
    print(f"🔥 Running main benchmark ({total_requests} requests)...\n")
    t_start = time.perf_counter()
    
    tasks = [asyncio.create_task(one_call(i)) for i in range(total_requests)]
    await asyncio.gather(*tasks)
    
    elapsed = time.perf_counter() - t_start
    
    # Calculate metrics
    latencies.sort()
    success_rate = ((total_requests - failures) / total_requests) * 100
    throughput = total_requests / elapsed
    
    p50 = latencies[int(0.50 * len(latencies)) - 1] if latencies else 0
    p95 = latencies[int(0.95 * len(latencies)) - 1] if latencies else 0
    p99 = latencies[int(0.99 * len(latencies)) - 1] if latencies else 0
    avg_latency = mean(latencies) if latencies else 0
    
    # Print results
    print(f"{'='*70}")
    print(f"📊 BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(f"\n🎯 THROUGHPUT:")
    print(f"   Measured: {throughput:.1f} req/s")
    print(f"   Target:   357.0 req/s")
    print(f"   Status:   {'✅ ACHIEVED' if throughput >= 357 else f'⚠️  {(throughput/357)*100:.1f}% of target'}")
    
    print(f"\n⏱️  LATENCY:")
    print(f"   Median (p50): {p50:.1f} ms")
    print(f"   P95:          {p95:.1f} ms")
    print(f"   P99:          {p99:.1f} ms")
    print(f"   Average:      {avg_latency:.1f} ms")
    
    print(f"\n✅ RELIABILITY:")
    print(f"   Success Rate: {success_rate:.2f}%")
    print(f"   Successful:   {total_requests - failures}")
    print(f"   Failed:       {failures}")
    
    print(f"\n⏳ TIMING:")
    print(f"   Total Time:   {elapsed:.2f} seconds")
    print(f"   Avg per req:  {(elapsed/total_requests)*1000:.2f} ms")
    
    print(f"\n{'='*70}\n")
    
    # Save results
    results = {
        "throughput": {
            "measured_req_per_sec": round(throughput, 1),
            "target_req_per_sec": 357.0,
            "achieved": throughput >= 357,
            "percentage_of_target": round((throughput/357)*100, 1)
        },
        "latency": {
            "median_ms": round(p50, 2),
            "p95_ms": round(p95, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(avg_latency, 2)
        },
        "reliability": {
            "success_rate_percent": round(success_rate, 2),
            "total_requests": total_requests,
            "successful_requests": total_requests - failures,
            "failed_requests": failures
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
        "comparison_to_sequential": {
            "sequential_baseline_req_per_sec": 61.1,
            "batched_async_req_per_sec": round(throughput, 1),
            "speedup_factor": round(throughput / 61.1, 2)
        }
    }
    
    with open('results/system_performance/batched_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: results/system_performance/batched_benchmark_results.json")
    
    return results

async def run_multiple_configs():
    """Test multiple concurrency configurations."""
    print("\n" + "="*70)
    print("🔬 MULTI-CONFIGURATION BENCHMARK")
    print("="*70 + "\n")
    
    configs = [
        (5000, 128),   # Lower concurrency
        (10000, 256),  # Target config
        (10000, 512),  # Higher concurrency
    ]
    
    all_results = []
    
    for total_reqs, concurrency in configs:
        print(f"\n📌 Testing: {total_reqs} requests @ {concurrency} concurrent")
        orchestrator = ModelOrchestrator()
        await orchestrator.initialize_batchers()
        
        result = await run_load_test(orchestrator, total_reqs, concurrency)
        all_results.append({
            "config": {"requests": total_reqs, "concurrency": concurrency},
            "throughput": result["throughput"]["measured_req_per_sec"]
        })
        
        # Brief pause between configs
        await asyncio.sleep(2)
    
    # Summary
    print("\n" + "="*70)
    print("📈 CONFIGURATION COMPARISON")
    print("="*70)
    for res in all_results:
        cfg = res["config"]
        tput = res["throughput"]
        status = "✅" if tput >= 357 else "⚠️ "
        print(f"{status} {cfg['requests']:5d} reqs @ {cfg['concurrency']:3d} concurrent → {tput:6.1f} req/s")
    print("="*70 + "\n")

async def main():
    """Main benchmark entry point."""
    print("\n🎯 SynchroChain Async + Batched Performance Benchmark")
    print("   Objective: Validate 357 req/s throughput target\n")
    
    # Initialize orchestrator
    print("⚙️  Initializing orchestrator with micro-batching...")
    orchestrator = ModelOrchestrator()
    await orchestrator.initialize_batchers()
    print("✅ Orchestrator ready\n")
    
    # Run comprehensive benchmark
    await run_load_test(orchestrator, total_requests=10000, concurrency=256)
    
    # Optional: run multiple configurations
    run_multi = input("\n🤔 Run multi-configuration test? (y/n): ").lower().strip()
    if run_multi == 'y':
        await run_multiple_configs()
    
    print("\n🎉 Benchmark completed!\n")

if __name__ == "__main__":
    asyncio.run(main())





