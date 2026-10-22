/*
 * Task 6: expose lost updates and compare them with an atomic increment.
 *
 * Build: clang -O2 -std=c11 -pthread task6.c -o task6
 */

#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <time.h>

#define STUDENT_ID 230103024
#define THREAD_COUNT 4
#define INCREMENTS_PER_THREAD UINT64_C(2000000)

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    size_t ready_threads;
    bool start;
} StartGate;

typedef struct {
    StartGate *gate;
    bool correct_atomic_increment;
} WorkerArguments;

typedef struct {
    uint64_t final_value;
    double elapsed_ms;
} BenchmarkResult;

static _Atomic uint64_t shared_counter;

static void fail(const char *message, int error_number) {
    errno = error_number;
    perror(message);
    exit(EXIT_FAILURE);
}

static double milliseconds_between(struct timespec start, struct timespec end) {
    time_t seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    return (double)seconds * 1000.0 + (double)nanoseconds / 1000000.0;
}

static void *increment_counter(void *raw_arguments) {
    WorkerArguments *arguments = raw_arguments;

    int error = pthread_mutex_lock(&arguments->gate->mutex);
    if (error != 0) {
        fail("pthread_mutex_lock", error);
    }
    arguments->gate->ready_threads++;
    pthread_cond_broadcast(&arguments->gate->condition);
    while (!arguments->gate->start) {
        error = pthread_cond_wait(
            &arguments->gate->condition,
            &arguments->gate->mutex
        );
        if (error != 0) {
            fail("pthread_cond_wait", error);
        }
    }
    pthread_mutex_unlock(&arguments->gate->mutex);

    if (arguments->correct_atomic_increment) {
        for (uint64_t iteration = 0; iteration < INCREMENTS_PER_THREAD; ++iteration) {
            atomic_fetch_add_explicit(&shared_counter, 1, memory_order_relaxed);
        }
    } else {
        /*
         * The load and store are individually well-defined, but together they
         * are deliberately not an atomic read-modify-write. Another thread can
         * overwrite an update between these two operations.
         */
        for (uint64_t iteration = 0; iteration < INCREMENTS_PER_THREAD; ++iteration) {
            uint64_t observed = atomic_load_explicit(
                &shared_counter,
                memory_order_relaxed
            );
            atomic_store_explicit(
                &shared_counter,
                observed + 1,
                memory_order_relaxed
            );
        }
    }

    return NULL;
}

static BenchmarkResult run_benchmark(bool correct_atomic_increment) {
    atomic_store_explicit(&shared_counter, 0, memory_order_relaxed);

    StartGate gate = {
        .mutex = PTHREAD_MUTEX_INITIALIZER,
        .condition = PTHREAD_COND_INITIALIZER,
        .ready_threads = 0,
        .start = false,
    };
    pthread_t threads[THREAD_COUNT];
    WorkerArguments arguments[THREAD_COUNT];

    for (size_t index = 0; index < THREAD_COUNT; ++index) {
        arguments[index] = (WorkerArguments){
            .gate = &gate,
            .correct_atomic_increment = correct_atomic_increment,
        };
        int error = pthread_create(
            &threads[index],
            NULL,
            increment_counter,
            &arguments[index]
        );
        if (error != 0) {
            fail("pthread_create", error);
        }
    }

    int error = pthread_mutex_lock(&gate.mutex);
    if (error != 0) {
        fail("pthread_mutex_lock", error);
    }
    while (gate.ready_threads != THREAD_COUNT) {
        error = pthread_cond_wait(&gate.condition, &gate.mutex);
        if (error != 0) {
            fail("pthread_cond_wait", error);
        }
    }

    struct timespec start;
    struct timespec end;
    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        fail("clock_gettime", errno);
    }
    gate.start = true;
    pthread_cond_broadcast(&gate.condition);
    pthread_mutex_unlock(&gate.mutex);

    for (size_t index = 0; index < THREAD_COUNT; ++index) {
        error = pthread_join(threads[index], NULL);
        if (error != 0) {
            fail("pthread_join", error);
        }
    }
    if (clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
        fail("clock_gettime", errno);
    }

    uint64_t final_value = atomic_load_explicit(
        &shared_counter,
        memory_order_relaxed
    );

    pthread_mutex_destroy(&gate.mutex);
    pthread_cond_destroy(&gate.condition);
    return (BenchmarkResult){
        .final_value = final_value,
        .elapsed_ms = milliseconds_between(start, end),
    };
}

int main(void) {
    const uint64_t expected = THREAD_COUNT * INCREMENTS_PER_THREAD;
    BenchmarkResult unsafe = run_benchmark(false);
    BenchmarkResult correct = run_benchmark(true);

    if (unsafe.final_value > expected) {
        fprintf(stderr, "Unsafe counter unexpectedly exceeded the target.\n");
        return EXIT_FAILURE;
    }
    if (correct.final_value != expected) {
        fprintf(stderr, "Atomic counter validation failed.\n");
        return EXIT_FAILURE;
    }

    uint64_t updates_lost = expected - unsafe.final_value;
    double lost_percentage = 100.0 * (double)updates_lost / (double)expected;
    double penalty_ms = correct.elapsed_ms - unsafe.elapsed_ms;
    double penalty_ratio = correct.elapsed_ms / unsafe.elapsed_ms;

    printf("Student ID: %d\n", STUDENT_ID);
    printf("Threads: %d\n", THREAD_COUNT);
    printf("Increments per thread: %" PRIu64 "\n", INCREMENTS_PER_THREAD);
    printf("Expected final value: %" PRIu64 "\n", expected);
    printf("Actual final value (unsynchronized): %" PRIu64 "\n", unsafe.final_value);
    printf("Updates lost: %" PRIu64 " (%.3f%%)\n", updates_lost, lost_percentage);
    printf("Unsynchronized wall time: %.3f ms\n", unsafe.elapsed_ms);
    printf("Atomic final value: %" PRIu64 "\n", correct.final_value);
    printf("Atomic wall time: %.3f ms\n", correct.elapsed_ms);
    printf(
        "Atomic synchronization penalty: %.3f ms (%.3fx slower)\n",
        penalty_ms,
        penalty_ratio
    );

    return EXIT_SUCCESS;
}
