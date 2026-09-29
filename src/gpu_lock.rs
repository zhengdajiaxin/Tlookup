#[cfg(feature = "icicle-rust")]
use std::{
    any::Any,
    cell::Cell,
    marker::PhantomData,
    panic::{self, AssertUnwindSafe},
    sync::{mpsc, OnceLock},
    thread,
};

#[cfg(feature = "icicle-rust")]
thread_local! {
    static GPU_WORKER_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[cfg(feature = "icicle-rust")]
struct GpuWorkerDepthGuard;

#[cfg(feature = "icicle-rust")]
impl GpuWorkerDepthGuard {
    fn enter() -> Self {
        GPU_WORKER_DEPTH.with(|depth| depth.set(depth.get() + 1));
        Self
    }
}

#[cfg(feature = "icicle-rust")]
impl Drop for GpuWorkerDepthGuard {
    fn drop(&mut self) {
        GPU_WORKER_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
    }
}

#[cfg(feature = "icicle-rust")]
struct GpuJob {
    run: unsafe fn(*mut ()),
    data: *mut (),
    done: mpsc::SyncSender<()>,
}

#[cfg(feature = "icicle-rust")]
unsafe impl Send for GpuJob {}

#[cfg(feature = "icicle-rust")]
struct GpuWorker {
    sender: mpsc::Sender<GpuJob>,
}

#[cfg(feature = "icicle-rust")]
static GPU_WORKER: OnceLock<GpuWorker> = OnceLock::new();

#[cfg(feature = "icicle-rust")]
pub(crate) fn with_gpu_lock<T, F>(f: F) -> T
where
    T: Send,
    F: FnOnce() -> T + Send,
{
    if GPU_WORKER_DEPTH.with(|depth| depth.get() > 0) {
        return f();
    }

    let worker = GPU_WORKER.get_or_init(GpuWorker::start);
    let mut state = GpuJobState {
        func: Some(f),
        result: None,
        _not_static: PhantomData,
    };
    let (done, finished) = mpsc::sync_channel(0);
    let job = GpuJob {
        run: run_gpu_job::<F, T>,
        data: (&mut state as *mut GpuJobState<'_, F, T>).cast::<()>(),
        done,
    };

    worker
        .sender
        .send(job)
        .expect("GPU worker thread is not running");
    finished.recv().expect("GPU worker dropped a submitted job");

    match state
        .result
        .take()
        .expect("GPU worker did not set a result")
    {
        Ok(value) => value,
        Err(payload) => panic::resume_unwind(payload),
    }
}

#[cfg(feature = "icicle-rust")]
impl GpuWorker {
    fn start() -> Self {
        let (sender, receiver) = mpsc::channel::<GpuJob>();
        thread::Builder::new()
            .name("fri-lookup-icicle-worker".to_string())
            .spawn(move || {
                for job in receiver {
                    let _depth = GpuWorkerDepthGuard::enter();
                    unsafe { (job.run)(job.data) };
                    let _ = job.done.send(());
                }
            })
            .expect("failed to start ICICLE GPU worker thread");
        Self { sender }
    }
}

#[cfg(feature = "icicle-rust")]
struct GpuJobState<'a, F, T> {
    func: Option<F>,
    result: Option<Result<T, Box<dyn Any + Send + 'static>>>,
    _not_static: PhantomData<&'a mut ()>,
}

#[cfg(feature = "icicle-rust")]
unsafe fn run_gpu_job<F, T>(data: *mut ())
where
    F: FnOnce() -> T,
{
    let state = unsafe { &mut *data.cast::<GpuJobState<'_, F, T>>() };
    let func = state.func.take().expect("GPU worker job already consumed");
    state.result = Some(panic::catch_unwind(AssertUnwindSafe(func)));
}

#[cfg(not(feature = "icicle-rust"))]
#[allow(dead_code)]
pub(crate) fn with_gpu_lock<T>(f: impl FnOnce() -> T) -> T {
    f()
}
