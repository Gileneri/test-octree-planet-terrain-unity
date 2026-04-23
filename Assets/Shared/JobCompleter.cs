using System;
using Unity.Jobs;

/// <summary>
/// Holds the three callbacks needed to manage a deferred mesh job:
///   schedule   — schedules the Burst job and returns its handle
///   onComplete — applies the finished mesh to the GameObject (main thread)
///   cancel     — called when the job is dropped by the budget system;
///                resets the node so it will be re-queued next frame
/// </summary>
public class JobCompleter
{
    public Func<JobHandle> schedule;
    public Action onComplete;
    public Action cancel;      // NEW — safe discard

    public JobCompleter(Func<JobHandle> schedule, Action onComplete, Action cancel = null)
    {
        this.schedule = schedule;
        this.onComplete = onComplete;
        this.cancel = cancel;
    }
}