#include <iostream>
#include <atomic>
#include <algorithm>
#include "pthread.h"
#include "MapReduceFramework.h"
#include "Barrier.cpp"

#define getInter (PrivateJobHandle*) (context->)

typedef struct PrivateJobHandle {
    const MapReduceClient& client;
    int numOfThreads;
    JobState* state;
    Barrier* barrier;
    InputVec inputVec;
    IntermediateVec* intermediateVec;
    OutputVec outputVec;
} PrivateJobHandle;

typedef struct ThreadContext {
    int threadID;
    std::atomic<uint32_t>* atomic_counter;
    PrivateJobHandle* jobHandler;
} ThreadContext;

std::vector<K2*> shuffle(std::vector<std::vector<V2*>> shuffled_vec);

PrivateJobHandle* jobHandle;
std::vector<ThreadContext> contexts;

bool cmp(const IntermediatePair &a, const IntermediatePair &b) {
    return a.first < b.first;
}

void* mapReduceFunc(void *arg) {
    auto* context = (ThreadContext*) arg;
    std::vector<std::vector<V2*>> shuffled_vec;
    for(auto pair: context->jobHandler->inputVec) {
        jobHandle->client.map(pair.first, pair.second, arg);
    }
    std::sort(context->jobHandler->intermediateVec->begin(), context->jobHandler->intermediateVec->end(), cmp);
    context->jobHandler->barrier->barrier();
    if (context->threadID == 0) {
        auto keys = shuffle(shuffled_vec);
    }
    for(auto vec: shuffled_vec) {
        jobHandle->client.reduce(vec, arg);
    }
}

std::vector<K2*> shuffle(std::vector<std::vector<V2*>> shuffled_vec) {
    //the first thread's intermediateVec
    int counter = jobHandle->numOfThreads;
    std::vector<K2*> keys;
    while (counter > 0) {
        IntermediateVec inter = *(contexts[counter - 1].jobHandler->intermediateVec);
        IntermediatePair pair = inter[inter.size() - 1];
        K2* maxKey = pair.first;
        for(const auto& context: contexts) {
            IntermediateVec interContext = *(context.jobHandler->intermediateVec);
            IntermediatePair pairContext = interContext[interContext.size() - 1];
            if (pairContext.first > maxKey) {
                maxKey = pairContext.first;
            }
        }
        keys.push_back(maxKey);
        std::vector<V2*> curVal;
        for(const auto& context: contexts) {
            IntermediateVec interContext = *(context.jobHandler->intermediateVec);
            IntermediatePair pairContext = interContext[interContext.size() - 1];
            if (pairContext.first == maxKey) {
                curVal.push_back(pairContext.second);
                interContext.pop_back();
                if (interContext.empty()) {
                    counter--;
                }
            }
        }
        shuffled_vec.push_back(curVal);
    }
    return keys;
};

JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel) {
    pthread_t threads[multiThreadLevel];
    ThreadContext contexts[multiThreadLevel];
    std::atomic<uint32_t> atomic_counter(0);
    JobState state = {UNDEFINED_STAGE, 0};
    jobHandle = new PrivateJobHandle({client, multiThreadLevel,
                                      std::vector<ThreadContext>(contexts, contexts + sizeof(contexts) / sizeof(contexts[0])), &state, });
    Barrier barrier(multiThreadLevel);
    auto* intermediateVec = new IntermediateVec();

    for (int i = 0; i < multiThreadLevel; ++i) {
        contexts[i] = {i, &atomic_counter, &barrier, inputVec, intermediateVec, outputVec};
    }

    for (int i = 0; i < multiThreadLevel; ++i) {
        if (pthread_create(threads + i, NULL, mapReduceFunc, contexts + i) != 0) {
            std::cerr << "system error: pthread create failed" << std::endl;
            exit(1);
        }
    }

    for (int i = 0; i < multiThreadLevel; ++i) {
        pthread_join(threads[i], NULL);
    }

    return (JobHandle) jobHandle;
}

void getJobState(JobHandle job, JobState* state) {
    auto* curJob = (PrivateJobHandle*) job;
    curJob->state = state;
}

void emit2 (K2* key, V2* value, void* context) {
    //todo: udate atopmic_var
    auto* tc = (ThreadContext*) context;
    IntermediateVec vec = *(tc->intermediateVec);
    vec.push_back(std::pair<K2*, V2*>(key, value));
}

void emit3 (K3* key, V3* value, void* context) {
    //todo: udate atopmic_var
    auto* tc = (ThreadContext*) context;
    tc->outputVec.push_back(std::pair<K3*, V3*>(key, value));
}