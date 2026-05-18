// ACORN-1 implementation is based on Lucene's Java implementation:
// https://github.com/apache/lucene/blob/main/lucene/core/src/java/org/apache/lucene/util/hnsw/FilteredHnswGraphSearcher.java
// Licensed under the Apache License, Version 2.0.

#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <mutex>
#include <random>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <unordered_set>
#include <list>
#include <type_traits>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

class NoopTerminationState {
public:
    inline void reset() {}
    inline void setEnabled( bool enabled ) {}
    inline void onDistanceScored() {}
    inline void onCandidateCollected() {}
    inline bool shouldTerminate(size_t ef, size_t current_size) { return false; }
};

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;
    static constexpr int DEFAULT_FILTERED_SEARCH_THRESHOLD = 60;
    static constexpr float ACORN_EXPANDED_EXPLORATION_LAMBDA = 0.10f;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  // number of deleted elements
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{ 0 };
    int filtered_search_threshold_{DEFAULT_FILTERED_SEARCH_THRESHOLD};

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    VisitedListPool *visited_list_pool_{nullptr};

    // Locks operations with element by label value
    // label_op_locks_ is upstream hnswlib's per-label-hash-bucket mutex set, used to serialize
    // markDelete/unmarkDelete/addPoint-with-replacement against each other. Manticore's build
    // path never deletes or replaces (every row gets a unique label), so this lock layer adds
    // overhead with no benefit.
//    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};

    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };

    char *data_level0_memory_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  // lock for label_lookup_
    std::unordered_map<labeltype, tableint> label_lookup_;

    // level_generator_ is consumed by getRandomLevel() in every addPoint(); std::default_random_engine
    // is not thread-safe, so concurrent addPoints would race the RNG state.
    std::default_random_engine level_generator_;
    mutable std::mutex level_generator_lock_;
    // update_probability_generator_ is only used by updatePoint() (delete/replace path). Not
    // protected: manticore's build never reaches that path.
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  // flag to replace deleted elements (marked as deleted) during insertions

//    std::mutex deleted_elements_lock;  // lock for deleted_elements
    std::unordered_set<tableint> deleted_elements;  // contains internal ids of deleted elements


    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }


    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : link_list_locks_(max_elements),
            //label_op_locks_(MAX_LABEL_OPERATION_LOCKS),  // disabled: delete/replace path not used
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();
        M_ = M;
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = new VisitedListPool(1, max_elements);

        // initializations for special treatment of the first node
        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        memset(linkLists_, 0, sizeof(void *) * max_elements_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }

    void freeLinkLists() {
        if (linkLists_ != nullptr) {
            if (element_levels_.size() > 0) {
                size_t iCount = element_levels_.size();
                if (cur_element_count < element_levels_.size())
                    iCount = cur_element_count;

                for (tableint i = 0; i < iCount; i++) {
                    if (element_levels_[i] > 0 && linkLists_[i] != nullptr) {
                        free(linkLists_[i]);
                    }
                }
            }
            free(linkLists_);
            linkLists_ = nullptr;
        }
    }

    ~HierarchicalNSW() {
        if ( data_level0_memory_!=nullptr )
            free ( data_level0_memory_ );
        
        freeLinkLists();
        
        if ( visited_list_pool_!=nullptr )
            delete visited_list_pool_;
    }


    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };
    using CandidatePair_t = std::pair<dist_t, tableint>;
    using CandidateQueue_t = std::priority_queue<CandidatePair_t, std::vector<CandidatePair_t>, CompareByFirst>;


    void setEf(size_t ef) {
        ef_ = ef;
    }

    void setFilteredSearchThreshold ( int filtered_search_threshold )
    {
        if ( filtered_search_threshold < 0 || filtered_search_threshold > 100 )
            throw std::runtime_error("filtered_search_threshold must be between 0 and 100");

        filtered_search_threshold_ = filtered_search_threshold;
    }

/*    inline std::mutex& getLabelOpMutex(labeltype label) const {
        // calculate hash
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }*/


    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

    template <class DistFn = void>
	inline dist_t calcDistance ( const void * query_data, tableint internal_id ) const
	{
	    if constexpr ( std::is_same_v<DistFn, void> )
	    {
	        return fstdistfunc_(
	            query_data,
	            getDataByInternalId(internal_id),
	            (labeltype)-1,
	            getExternalLabel(internal_id),
	            dist_func_param_ );
	    }
	    else
	    {
	        return DistFn::Eval(
	            query_data,
	            getDataByInternalId(internal_id),
	            (size_t)-1,
	            internal_id,
	            dist_func_param_ );
	    }
	}

    template <class DistFn = void>
    inline void calcDistance2 ( const void * query_data, tableint internal_id_a, tableint internal_id_b, dist_t & dist_a, dist_t & dist_b ) const
    {
        if constexpr ( std::is_same_v<DistFn, void> )
        {
            dist_a = fstdistfunc_(
                query_data,
                getDataByInternalId(internal_id_a),
                (labeltype)-1,
                getExternalLabel(internal_id_a),
                dist_func_param_ );
            dist_b = fstdistfunc_(
                query_data,
                getDataByInternalId(internal_id_b),
                (labeltype)-1,
                getExternalLabel(internal_id_b),
                dist_func_param_ );
        }
        else
        {
            DistFn::Eval2(
                query_data,
                getDataByInternalId(internal_id_a),
                getDataByInternalId(internal_id_b),
                (size_t)-1,
                internal_id_a,
                internal_id_b,
                dist_func_param_,
                dist_a,
                dist_b );
        }
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }


    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }


    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }


    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double sample;
        {
            std::lock_guard<std::mutex> lock(level_generator_lock_);
            sample = distribution(level_generator_);
        }
        double r = -log(sample) * reverse_size;
        return (int) r;
    }

    size_t getMaxElements() {
        return max_elements_;
    }

    size_t getCurrentElementCount() {
        return cur_element_count;
    }

    size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer, labeltype label) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;
        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), label, getExternalLabel(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }
        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  // = (int *)(linkList0_ + curNodeNum * size_links_per_element0_);
            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
//                    data = (int *) (linkLists_[curNodeNum] + (layer - 1) * size_links_per_element_);
            }
            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
//                    if (candidate_id == 0) continue;
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, label, getExternalLabel(candidate_id), dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);

                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();

                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }


    float estimateFilterRatio ( BaseFilterFunctor * isIdAllowed ) const
    {
        if ( !isIdAllowed )
            return 1.0f;

        size_t total = cur_element_count.load();
        if ( !total )
            return 0.0f;

        long long rawCount = isIdAllowed->getFilterCount();
        if ( rawCount >= 0 )
        {
            size_t count = rawCount;
            if ( count > total )
                count = total;

            return float(count) / total;
        }

        return 1.0f;
    }

    bool shouldUseAcorn ( BaseFilterFunctor * isIdAllowed ) const
    {
        if ( !isIdAllowed || filtered_search_threshold_ <= 0 || isIdAllowed->getFilterCount() < 0)
            return false;

        float ratio = estimateFilterRatio(isIdAllowed);
        if ( ratio <= 0.0f )
            return false;

        return ratio * 100.0f < filtered_search_threshold_;
    }

    bool shouldBypassHnswForFilteredSearch ( size_t k, long long iFilterCount, size_t ef ) const
    {
        if ( iFilterCount < 0 )
            return false;

        size_t graphSize = cur_element_count.load();
        if ( !graphSize )
            return true;

        size_t filteredCount = iFilterCount;
        if ( filteredCount > graphSize )
            filteredCount = graphSize;

        if ( !filteredCount )
            return true;

        size_t searchEf = std::max ( std::max(ef_, k), ef );

        bool doHnsw = k < filteredCount;
        int unfilteredVisit = expectedVisitedNodes ( searchEf, graphSize );
        if ( (size_t)unfilteredVisit >= filteredCount )
            doHnsw = false;

        return !doHnsw;
    }

    template <typename TerminationPolicy, class AllowTopCandidateFn>
    inline void processScoredCandidate (
        CandidateQueue_t & top_candidates,
        CandidateQueue_t & candidate_set,
        dist_t & lowerBound,
        size_t ef,
        tableint candidate_id,
        dist_t dist,
        TerminationPolicy & termination_state,
        const AllowTopCandidateFn & fnAllowTopCandidate ) const
    {
        termination_state.onDistanceScored();

        if (top_candidates.size() < ef || lowerBound > dist) {
            candidate_set.emplace(-dist, candidate_id);

            if (fnAllowTopCandidate(candidate_id)) {
                top_candidates.emplace(dist, candidate_id);
                termination_state.onCandidateCollected();
            }

            if (top_candidates.size() > ef)
                top_candidates.pop();

            if (!top_candidates.empty())
                lowerBound = top_candidates.top().first;
        }
    }

    template <typename TerminationPolicy, bool collect_metrics = false, class DistFn = void, class AllowTopCandidateFn>
    inline void searchBaseLayerPass12 (
        const void * data_point,
        size_t ef,
        vl_type * visited_array,
        vl_type visited_array_tag,
        CandidateQueue_t & top_candidates,
        CandidateQueue_t & candidate_set,
        dist_t & lowerBound,
        TerminationPolicy & termination_state,
        const AllowTopCandidateFn & fnAllowTopCandidate ) const
    {
        tableint dLocalBatch[64];
        std::vector<tableint> dBatchOverflow;

        while (!candidate_set.empty()) {
            CandidatePair_t current_node_pair = candidate_set.top();

            if ((-current_node_pair.first) > lowerBound && top_candidates.size() == ef)
                break;

            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            assert ( size <= maxM0_ );
            if constexpr (collect_metrics) {
                metric_hops++;
                metric_distance_computations += size;
            }

#ifdef USE_SSE
            if (!candidate_set.empty()) {
                tableint next_node_id = candidate_set.top().second;
                _mm_prefetch(data_level0_memory_ + next_node_id * size_data_per_element_ + offsetLevel0_, _MM_HINT_T0);
            }
#endif

            tableint * pBatch = dLocalBatch;
            if ( size > std::size(dLocalBatch) )
            {
                dBatchOverflow.resize(size);
                pBatch = dBatchOverflow.data();
            }
            size_t iBatchSize = 0;

            // Pass 1: prefetch visited slots, then gather unvisited neighbors and prefetch their vector data.
#ifdef USE_SSE
            for (size_t j = 1; j <= size; j++)
                _mm_prefetch((char *) (visited_array + *(data + j)), _MM_HINT_T0);
#endif
            for (size_t j = 1; j <= size; j++) {
                tableint candidate_id = *(data + j);
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;
                    pBatch[iBatchSize++] = candidate_id;
#ifdef USE_SSE
                    _mm_prefetch(data_level0_memory_ + (size_t)candidate_id * size_data_per_element_ + offsetData_, _MM_HINT_T0);
#endif
                }
            }

            // Pass 2: compute distances + update heaps (same order as gathered)
            size_t b = 0;
            for ( ; b + 1 < iBatchSize; b += 2 ) {
                dist_t distA, distB;
                calcDistance2<DistFn>(data_point, pBatch[b], pBatch[b + 1], distA, distB);
                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, pBatch[b], distA, termination_state, fnAllowTopCandidate);

                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, pBatch[b + 1], distB, termination_state, fnAllowTopCandidate);
            }

            for ( ; b < iBatchSize; b++ ) {
                dist_t dist = calcDistance<DistFn>(data_point, pBatch[b]);
                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, pBatch[b], dist, termination_state, fnAllowTopCandidate);
            }

            if ( termination_state.shouldTerminate(ef, top_candidates.size()) )
                break;
        }
    }

    template <typename TerminationPolicy = NoopTerminationState, bool has_deletions, bool collect_metrics = false, class DistFn = void>
    CandidateQueue_t
    searchBaseLayerSTNoFilter(tableint ep_id, const void *data_point, size_t ef) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        CandidateQueue_t top_candidates;
        CandidateQueue_t candidate_set;

        dist_t lowerBound;
        dist_t dist = calcDistance<DistFn> ( data_point, ep_id );
        lowerBound = dist;
        top_candidates.emplace(dist, ep_id);
        candidate_set.emplace(-dist, ep_id);

        visited_array[ep_id] = visited_array_tag;

        TerminationPolicy termination_state;
        auto fnAllowTopCandidate = [] ( tableint ) { return true; };
        searchBaseLayerPass12<TerminationPolicy, collect_metrics, DistFn> (
            data_point, ef, visited_array, visited_array_tag, top_candidates, candidate_set, lowerBound, termination_state, fnAllowTopCandidate );

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <typename TerminationPolicy = NoopTerminationState, bool has_deletions, bool collect_metrics = false, class DistFn = void>
    CandidateQueue_t
    searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef, BaseFilterFunctor* isIdAllowed = nullptr) const {
        if ( shouldUseAcorn(isIdAllowed) )
            return searchBaseLayerSTFilteredAcorn<TerminationPolicy, has_deletions, collect_metrics, DistFn>(ep_id, data_point, ef, isIdAllowed);

        if constexpr (!has_deletions) {
            if (!isIdAllowed)
                return searchBaseLayerSTNoFilter<TerminationPolicy, has_deletions, collect_metrics, DistFn>(ep_id, data_point, ef);
        }

        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        CandidateQueue_t top_candidates;
        CandidateQueue_t candidate_set;

        dist_t lowerBound;
        if ((!has_deletions || !isMarkedDeleted(ep_id)) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id)))) {
            dist_t dist = calcDistance<DistFn> ( data_point, ep_id );
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        TerminationPolicy termination_state;
        auto fnAllowTopCandidate = [&] ( tableint candidate_id )
        {
            return (!has_deletions || !isMarkedDeleted(candidate_id))
                && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id)));
        };
        searchBaseLayerPass12<TerminationPolicy, collect_metrics, DistFn> (
            data_point, ef, visited_array, visited_array_tag, top_candidates, candidate_set, lowerBound, termination_state, fnAllowTopCandidate );

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <typename TerminationPolicy = NoopTerminationState, bool has_deletions, bool collect_metrics = false, class DistFn = void>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerSTFilteredAcorn(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed
    ) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        auto isAllowed = [&](tableint id) -> bool
        {
            return isIdAllowed == nullptr || (*isIdAllowed)(getExternalLabel(id));
        };

        dist_t lowerBound = std::numeric_limits<dist_t>::max();
        dist_t dist = calcDistance<DistFn> ( data_point, ep_id );
        candidate_set.emplace(-dist, ep_id);
        if ((!has_deletions || !isMarkedDeleted(ep_id)) && isAllowed(ep_id))
        {
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        float filter_ratio = estimateFilterRatio(isIdAllowed);
        float min_ratio = 1.0f / std::max<size_t>(cur_element_count.load(), 1);
        if (filter_ratio < min_ratio)
            filter_ratio = min_ratio;

        float inverse_ratio = 1.0f / filter_ratio;
        int maxExplorationMultiplier = (int)std::min(inverse_ratio, (float)maxM0_ / 2.0f);
        if (maxExplorationMultiplier < 1)
            maxExplorationMultiplier = 1;

        int minToScore = (int)std::min(std::max(0.0f, inverse_ratio - (2.0f * maxM0_)), (float)maxM0_);

        std::vector<tableint> toScore;
        std::vector<tableint> toExplore;
        size_t queue_capacity = maxM0_ * 2 * maxExplorationMultiplier;
        toScore.reserve(queue_capacity);
        toExplore.reserve(queue_capacity);

        TerminationPolicy termination_state;
        auto fnAllowTopCandidate = [&] ( tableint candidate_id )
        {
            if constexpr (has_deletions)
                if (isMarkedDeleted(candidate_id))
                    return false;
            return isAllowed(candidate_id);
        };

        while ( !candidate_set.empty() )
        {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            if ((-current_node_pair.first) > lowerBound && top_candidates.size() == ef)
                break;

            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);

            if (collect_metrics)
                metric_hops++;

            toScore.clear();
            toExplore.clear();
            size_t exploreUpto = 0;

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif
            for (size_t j = 1; j <= size; j++)
            {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag)
                    continue;

                visited_array[candidate_id] = visited_array_tag;

                bool allowed = isAllowed(candidate_id);
                bool deleted = has_deletions && isMarkedDeleted(candidate_id);
                if (allowed || deleted)
                    toScore.push_back(candidate_id);
                else
                    toExplore.push_back(candidate_id);
            }

            if ( !toExplore.empty() )
            {
                float filteredAmount = size == 0 ? 0.0f : (float)toExplore.size() / size;
                float denom = std::max(1e-6f, 1.0f - filteredAmount);
                int maxToScoreCount = int(size * std::min((float)maxExplorationMultiplier, 1.0f / denom) );

                if ( (int)toScore.size() < maxToScoreCount && filteredAmount > ACORN_EXPANDED_EXPLORATION_LAMBDA )
                {
                    size_t maxAdditionalToExploreCount = toExplore.capacity() > 0 ? (toExplore.capacity() - 1) : 0;
                    size_t totalExplored = toScore.size() + toExplore.size();
                    while (exploreUpto < toExplore.size()
                        && totalExplored < maxAdditionalToExploreCount
                        && (int)toScore.size() < maxToScoreCount)
                    {
                        tableint exploreNeighbor = toExplore[exploreUpto++];
                        int *exploreData = (int *) get_linklist0(exploreNeighbor);
                        size_t exploreSize = getListCount((linklistsizeint*)exploreData);
#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *(exploreData + 1)), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *(exploreData + 1) + 64), _MM_HINT_T0);
                        _mm_prefetch(data_level0_memory_ + (*(exploreData + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
                        _mm_prefetch((char *) (exploreData + 2), _MM_HINT_T0);
#endif
                        for (size_t k = 1; k <= exploreSize && (int)toScore.size() < maxToScoreCount; k++)
                        {
                            int neighborOfNeighbor = *(exploreData + k);
#ifdef USE_SSE
                            _mm_prefetch((char *) (visited_array + *(exploreData + k + 1)), _MM_HINT_T0);
                            _mm_prefetch(data_level0_memory_ + (*(exploreData + k + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
#endif
                            if (visited_array[neighborOfNeighbor] == visited_array_tag)
                                continue;

                            visited_array[neighborOfNeighbor] = visited_array_tag;
                            totalExplored++;
                            bool allowed = isAllowed(neighborOfNeighbor);
                            bool deleted = has_deletions && isMarkedDeleted(neighborOfNeighbor);
                            if (allowed || deleted)
                                toScore.push_back(neighborOfNeighbor);
                            else if (totalExplored < maxAdditionalToExploreCount && (int)toScore.size() < minToScore)
                                toExplore.push_back(neighborOfNeighbor);
                        }
                    }
                }
            }

            if (collect_metrics)
                metric_distance_computations += toScore.size();

            size_t i = 0;
            for ( ; i + 1 < toScore.size(); i += 2 )
            {
                tableint candidate_id_a = toScore[i];
                tableint candidate_id_b = toScore[i + 1];
                dist_t distA, distB;
                calcDistance2<DistFn> ( data_point, candidate_id_a, candidate_id_b, distA, distB );
                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, candidate_id_a, distA, termination_state, fnAllowTopCandidate);
                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, candidate_id_b, distB, termination_state, fnAllowTopCandidate);
            }

            for ( ; i < toScore.size(); i++ )
            {
                tableint candidate_id = toScore[i];
                dist_t dist = calcDistance<DistFn> ( data_point, candidate_id );
                processScoredCandidate(top_candidates, candidate_set, lowerBound, ef, candidate_id, dist, termination_state, fnAllowTopCandidate);
            }

            if ( termination_state.shouldTerminate(ef, top_candidates.size()) )
                break;
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }


    void getNeighborsByHeuristic2(
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist = fstdistfunc_ ( getDataByInternalId(second_pair.second), getDataByInternalId(curent_pair.second), getExternalLabel(second_pair.second), getExternalLabel(curent_pair.second), dist_func_param_ );
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }


    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }


    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }


    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }


    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;
        getNeighborsByHeuristic2(top_candidates, M_);
        if (top_candidates.size() > M_)
            throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            // lock only during the update
            // because during the addition the lock for cur_c is already acquired
//            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
 //           if (isUpdate) {
//                lock.lock();
//            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }
            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                if (level > element_levels_[selectedNeighbors[idx]])
                    throw std::runtime_error("Trying to make a link on a non-existent level");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            if (sz_link_list_other > Mcurmax)
                throw std::runtime_error("Bad value of sz_link_list_other");
            if (selectedNeighbors[idx] == cur_c)
                throw std::runtime_error("Trying to connect an element to itself");
            if (level > element_levels_[selectedNeighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            // If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {
                    // finding the "weakest" element to replace it with the new one
                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]), getExternalLabel(cur_c), getExternalLabel(selectedNeighbors[idx]), dist_func_param_);
                    // Heuristic:
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]), getExternalLabel(data[j]), getExternalLabel(selectedNeighbors[idx]), dist_func_param_)
                            , data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                    // Nearest K:
                    /*int indx = -1;
                    for (int j = 0; j < sz_link_list_other; j++) {
                        dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
                        if (d > d_max) {
                            indx = j;
                            d_max = d;
                        }
                    }
                    if (indx >= 0) {
                        data[indx] = cur_c;
                    } */
                }
            }
        }

        return next_closest_entry_point;
    }


    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        delete visited_list_pool_;
        visited_list_pool_ = new VisitedListPool(1, new_max_elements);

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        // Reallocate base layer
        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        // Reallocate all other layers
        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }


    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

	template <typename WRITER>
	void saveIndex ( WRITER & tWriter )
	{
		tWriter.Write ( offsetLevel0_ );
		tWriter.Write ( max_elements_ );
        tWriter.Write ( cur_element_count );
        tWriter.Write ( size_data_per_element_ );
        tWriter.Write ( label_offset_ );
        tWriter.Write ( offsetData_ );
        tWriter.Write ( maxlevel_ );
        tWriter.Write ( enterpoint_node_);
        tWriter.Write ( maxM_ );

        tWriter.Write ( maxM0_ );
        tWriter.Write ( M_ );
        tWriter.Write ( mult_ );
        tWriter.Write ( ef_construction_ );

        tWriter.Write ( (uint8_t*)data_level0_memory_, cur_element_count * size_data_per_element_ );
        for (size_t i = 0; i < cur_element_count; i++)
		{
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            tWriter.Write ( linkListSize );
            if (linkListSize)
                tWriter.Write ( (uint8_t*)(linkLists_[i]), linkListSize );
        }
	}

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        // get file size:
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        /// Optional - check if index is ok:
        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        // throw exception if it either corrupted or old index
        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();
        /// Optional check end

        input.seekg(pos, input.beg);

        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
//        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);  // disabled: delete/replace path not used

        visited_list_pool_ = new VisitedListPool(1, max_elements);

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    template <typename READER>
    bool loadIndex ( READER & tReader, SpaceInterface<dist_t> *s, std::string & error )
    {
        size_t total_filesize = tReader.GetFileSize();

        tReader.Read ( offsetLevel0_ );
        tReader.Read ( max_elements_ );
        tReader.Read ( cur_element_count );

        tReader.Read ( size_data_per_element_ );
        tReader.Read ( label_offset_ );
        tReader.Read ( offsetData_ );
        tReader.Read ( maxlevel_ );
        tReader.Read ( enterpoint_node_ );

        tReader.Read ( maxM_ );
        tReader.Read ( maxM0_ );
        tReader.Read ( M_ );
        tReader.Read ( mult_ );
        tReader.Read ( ef_construction_ );

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = tReader.GetPos();

        /// Optional - check if index is ok:
        tReader.Seek ( pos + cur_element_count * size_data_per_element_ );
        for (size_t i = 0; i < cur_element_count; i++) {
            if ( tReader.GetPos() < 0 || tReader.GetPos() >= total_filesize )
            {
                error = "Index seems to be corrupted or unsupported";
                return false;
            }

            unsigned int linkListSize;
            tReader.Read ( linkListSize );
            if (linkListSize != 0)
                tReader.Seek ( tReader.GetPos() + linkListSize );
        }

        // throw exception if it either corrupted or old index
        if ( tReader.GetPos()>total_filesize )
        {
            error = "Index seems to be corrupted or unsupported";
            return false;

        }
        /// Optional check end

        tReader.Seek(pos);

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
        {
            error = "Not enough memory: loadIndex failed to allocate level0";
            return false;
        }

        tReader.Read ( (uint8_t*)data_level0_memory_, cur_element_count * size_data_per_element_ );

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements_).swap(link_list_locks_);
//        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);  // disabled: delete/replace path not used

        visited_list_pool_ = new VisitedListPool(1, max_elements_);

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
        {
            error = "Not enough memory: loadIndex failed to allocate linklists";
            return false;
        }
        memset(linkLists_, 0, sizeof(void *) * max_elements_);

        element_levels_ = std::vector<int>(max_elements_);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            tReader.Read(linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                {
                    error = "Not enough memory: loadIndex failed to allocate linklist";
                    return false;
                }
                tReader.Read ( (uint8_t*)(linkLists_[i]), linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        return true;
    }

    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        // lock all operations with element by label
//        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        
//        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
//        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (int i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }


    /*
    * Marks an element with the given label deleted, does NOT really change the current graph.
    */
    void markDelete(labeltype label) {
        // lock all operations with element by label
//        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

//        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
//        lock_table.unlock();

        markDeletedInternal(internalId);
    }


    /*
    * Uses the last 16 bits of the memory for the linked list size to store the mark,
    * whereas maxM0_ has to be limited to the lower 16 bits, however, still large enough in almost all cases.
    */
    void markDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
//                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }


    /*
    * Removes the deleted mark of the node, does NOT really change the current graph.
    * 
    * Note: the method is not safe to use when replacement of deleted elements is enabled,
    *  because elements marked as deleted can be completely removed by addPoint
    */
    void unmarkDelete(labeltype label) {
        // lock all operations with element by label
//        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

//        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found");
        }
        tableint internalId = search->second;
//        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }



    /*
    * Remove the deleted mark of the node.
    */
    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < cur_element_count);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
//                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }


    /*
    * Checks the first 16 bits of the memory to see if the element is marked deleted.
    */
    bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }


    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }


    /*
    * Adds point. Updates the point if it is already in the index.
    * If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point
    */
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        // lock all operations with element by label
        //std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        // check if there is vacant place
        tableint internal_id_replaced;
        //std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        //lock_deleted_elements.unlock();

        // if there is no vacant place then add or update point
        // else add point to vacant place
        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            // we assume that there are no concurrent operations on deleted element
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            //std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            //lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }


    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        // update the feature vector associated with existing point with new vector
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        // If point to be updated is entry point and graph just contains single element then just return.
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);

                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {
                // if (neigh == internalId)
                //     continue;

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  // sCand guaranteed to have size >= 1
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), getExternalLabel(neigh), getExternalLabel(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                // Retrieve neighbours using heuristic and set connections.
                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);

                {
//                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }


    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), getExternalLabel(dataPointInternalId), getExternalLabel(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
//                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), getExternalLabel(dataPointInternalId), getExternalLabel(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer ( currObj, dataPoint, level, getExternalLabel(dataPointInternalId) );

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }

            // Since element_levels_ is being used to get `dataPointLevel`, there could be cases where `topCandidates` could just contains entry point itself.
            // To prevent self loops, the `topCandidates` is filtered and thus can be empty.
            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), getExternalLabel(dataPointInternalId), getExternalLabel(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }


    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
//        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }


    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            // Checking if the element with the same label already exists
            // if so, updating it *instead* of creating a new element.
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        // Initialisation of the data and label
        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), label, getExternalLabel(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            if (cand < 0 || cand > max_elements_)
                                throw std::runtime_error("cand error");
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), label, getExternalLabel(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                if (level > maxlevelcopy || level < 0)  // possible?
                    throw std::runtime_error("Level error");

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer ( currObj, data_point, level, label );
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), (labeltype)-1, getExternalLabel(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            // Do nothing for the first element
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        // Releasing lock for the maximum level
        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::vector<std::pair<dist_t, labeltype>>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr,
        size_t * ef = nullptr) const override {
        return searchKnn<NoopTerminationState, false, void>(query_data, k, isIdAllowed, ef);
    }


    template <typename TerminationPolicy = NoopTerminationState, bool collect_metrics = false, class DistFn = void>
    std::vector<std::pair<dist_t, labeltype>>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr,
              size_t * ef = nullptr) const {
        std::vector<std::pair<dist_t, labeltype>> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = calcDistance<DistFn> ( query_data, enterpoint_node_ );

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                if constexpr (collect_metrics) {
                    metric_hops++;
                    metric_distance_computations += size;
                }

                tableint *datal = (tableint *) (data + 1);
                int i = 0;
                for ( ; i + 1 < size; i += 2 ) {
                    tableint candA = datal[i];
                    tableint candB = datal[i + 1];
                    if (candA < 0 || candA > max_elements_)
                        throw std::runtime_error("cand error");
                    if (candB < 0 || candB > max_elements_)
                        throw std::runtime_error("cand error");

                    dist_t dA, dB;
                    calcDistance2<DistFn> ( query_data, candA, candB, dA, dB );

                    if (dA < curdist) {
                        curdist = dA;
                        currObj = candA;
                        changed = true;
                    }

                    if (dB < curdist) {
                        curdist = dB;
                        currObj = candB;
                        changed = true;
                    }
                }

                for ( ; i < size; i++ ) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = calcDistance<DistFn> ( query_data, cand );

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        size_t searchEf = std::max(ef_, k);
        if ( ef )
            searchEf = std::max(searchEf, *ef);
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        if (num_deleted_) {
            top_candidates = searchBaseLayerST<TerminationPolicy, true, collect_metrics, DistFn>(
                    currObj, query_data, searchEf, isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<TerminationPolicy, false, collect_metrics, DistFn>(
                    currObj, query_data, searchEf, isIdAllowed);
        }

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        result.reserve(top_candidates.size());
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.emplace_back(rez.first, getExternalLabel(rez.second));
            top_candidates.pop();
        }
        return result;
    }


    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] > 0);
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

private:
    static int expectedVisitedNodes ( size_t k, size_t graphSize )
	{
        if ( !k || graphSize <= 1)
            return 0;

        return int(log((double)graphSize) * (double)std::min(k, graphSize));
    }
};
}  // namespace hnswlib
