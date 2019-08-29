/*
 * counter.cpp -- Archer runtime library, counter header file
 */
  
  //===----------------------------------------------------------------------===//
  //
  // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
  // See tools/archer/LICENSE.txt for details.
  // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
  //
  //===----------------------------------------------------------------------===//
  
  

#include <stdio.h>
#include <inttypes.h>
#include <omp-tools.h>

// 240 should be enough for xeon-phi
#define MAX_THREADS 240

#define CACHE_LINE 128

#define COUNT_EVENT1(name)                                                     \
  if (this_event_counter)                                                      \
  this_event_counter->name++
#define COUNT_EVENT2(name, scope)                                              \
  if (this_event_counter)                                                      \
  this_event_counter->name##_##scope++
#define COUNT_EVENT3(name, scope, kind)                                        \
  if (this_event_counter)                                                      \
  this_event_counter->name##_##scope##_##kind++

typedef struct alignas(128) {
  int thread_begin; // (1)	thread_begin
  int thread_end; // (2)	thread_end
  int parallel_begin; // (3)	parallel_begin
  int parallel_end; // (4)	parallel_end
  int task_create_initial; // (5)	task_create: 	task_initial
  int task_create_explicit; //			task_explicit
  int task_create_target; //			task_target
  int task_create_included; //			task_included
  int task_create_untied; //			task_untied
  int task_schedule; // (6)	task_schedule
  int implicit_task_scope_begin; // (7)	implicit task:	scope_begin
  int implicit_task_scope_end; //			scope_end
  int mutex_released_lock; // (15) mutex_released:	mutex_lock
  int mutex_released_nest_lock; //			mutex_nest_lock
  int mutex_released_critical; //			mutex_critical
  int mutex_released_atomic; //			mutex_atomic
  int mutex_released_ordered; //			mutex_ordered
  int mutex_released_default; //			default
  int task_dependences; // (16) task_dependences
  int task_dependence; // (17) task_dependence
  int sync_region_scope_begin_barrier; // (21) sync_region:	scope_begin:
                                       //                sync_region_barrier
  int sync_region_scope_begin_reduction; //              sync_region_reduction
  int sync_region_scope_begin_taskwait; //               sync_region_taskwait
  int sync_region_scope_begin_taskgroup; //              sync_region_taskgroup
  int sync_region_scope_end_barrier; //                  	scope_end:
                                     //                  sync_region_barrier
  int sync_region_scope_end_reduction; //                sync_region_reduction
  int sync_region_scope_end_taskwait; //                 sync_region_taskwait
  int sync_region_scope_end_taskgroup; //                sync_region_taskgroup
  int lock_init_lock; // (22) lock_init:	mutex_lock
  int lock_init_nest_lock; // 			mutex_nest_lock
  int lock_init_default; //			default
  int lock_destroy_lock; // (23) lock_destroy	mutex_lock
  int lock_destroy_nest_lock; //			mutex_nest_lock
  int lock_destroy_default; // 			default
  int mutex_acquire_lock; // (24) mutex_acquire:	mutex_lock
  int mutex_acquire_nest_lock; // 			mutex_nest_lock
  int mutex_acquire_critical; //			mutex_critical
  int mutex_acquire_atomic; // 			mutex_atomic
  int mutex_acquire_ordered; //			mutex_ordered
  int mutex_acquire_default; //			default
  int mutex_acquired_lock; // (25) mutex_acquired: mutex_lock
  int mutex_acquired_nest_lock; //                      mutex_nest_lock
  int mutex_acquired_critical; //                      mutex_critical
  int mutex_acquired_atomic; //                      mutex_atomic
  int mutex_acquired_ordered; //                      mutex_ordered
  int mutex_acquired_default; //			default
  int nest_lock_scope_begin; // (26) nest_lock:	scope_begin
  int nest_lock_scope_end; //			scope_end
  int flush; // (27) flush
} callback_counter_t;

#ifdef __cplusplus
extern "C" {
#endif

void print_callbacks(callback_counter_t *counter);

#ifdef __cplusplus
}
#endif
