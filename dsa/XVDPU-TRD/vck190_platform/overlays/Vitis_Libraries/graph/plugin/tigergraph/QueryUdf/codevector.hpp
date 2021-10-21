/*
 * File:   CodeVector.hpp
 * Author: dliddell
 *
 * Created on December 24, 2019, 5:08 PM
 */

#ifndef CODEVECTOR_HPP
#define CODEVECTOR_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

namespace xai {
class CodeToIdMap;
extern CodeToIdMap* pMap;
extern const unsigned int startPropertyIndex;
extern std::vector<uint64_t> IDMap;

typedef std::int32_t CosineVecValue; ///< A value for an element of a cosine similarity vector
typedef std::uint64_t SnomedCode;    ///< A SNOMED CT medical code
typedef unsigned SnomedId;           ///< SNOMED CT code as a small int
typedef unsigned SnomedConcept;      ///< Integer that distinguishes one concept from
                                     /// another (allergy vs. treatment, etc.)

// Standard values for cosine vector elements

#ifdef UNIT_TEST
const CosineVecValue MaxVecValue = 11000;
const CosineVecValue MinVecValue = 10000;
const CosineVecValue NullVecValue = -10000;
#else
const CosineVecValue MaxVecValue = 10480;
const CosineVecValue MinVecValue = MaxVecValue / 2;
const CosineVecValue NullVecValue = -MaxVecValue;
#endif

std::int64_t abs64(std::int64_t x);

/**
 * Holds a map from 64-bit SNOMED CT codes to smaller integers called "IDs"
 *
 * Call getInstance()->addCode(code) to add codes to the map.  The function
 * returns the ID if needed.
 * Call getInstance(true) to clean up the singleton.
 *
 */
class CodeToIdMap {
    typedef std::unordered_map<SnomedCode, SnomedId> Map;

    struct ConceptMap {
        Map m_map;
        SnomedId m_nextId = 0;
        SnomedId m_numReservedIds = 0;
    };

    std::vector<ConceptMap> m_ids;

   public:
    static CodeToIdMap* getInstance(bool release = false) {
        // static CodeToIdMap *pMap = nullptr;
        if (release)
            delete xai::pMap;
        else if (xai::pMap == nullptr)
            xai::pMap = new CodeToIdMap;
        return xai::pMap;
    }

    static void dump() {
        auto pInst = getInstance();
        std::cout << "CodeToIdMap::dump() instance=" << std::hex << pInst << std::dec << std::endl;
        std::cout << "  Num reserved ids: ";
        for (ConceptMap& map : pInst->m_ids) std::cout << map.m_numReservedIds << " ";
        std::cout << std::endl << std::flush;
    }

    static void reserveIds(SnomedConcept concept, SnomedId numIds) {
        getInstance()->reserveIdsInternal(concept, numIds);
        // dump();
    }

    SnomedId addCode(SnomedConcept concept, SnomedCode code) {
        // Get the map for the given concept, creating the map if it doesn't already
        // exist
        ConceptMap& map = getConceptMap(concept);

        // Try to insert the code into the map
        auto insertResult = map.m_map.insert({code, map.m_nextId});

        // Whatever the result of the insertion, get the small int associated with
        // the big int
        SnomedId id = insertResult.first->second;

        // If the insertion was successful, we used up a small int
        if (insertResult.second) {
            ++map.m_nextId;
            if (map.m_nextId > map.m_numReservedIds) map.m_numReservedIds = map.m_nextId;
        }

        return id;
    }

    SnomedId getNumReservedIds(SnomedConcept concept) const {
        // dump();
        if (concept >= m_ids.size()) return 0;
        return m_ids[concept].m_numReservedIds;
    }

   private:
    void reserveIdsInternal(SnomedConcept concept, SnomedId numIds) {
        // Get the map for the given concept, creating the map if it doesn't already
        // exist
        ConceptMap& map = getConceptMap(concept);

        if (numIds > map.m_numReservedIds) map.m_numReservedIds = numIds;
    }

    void reserveConcept(SnomedConcept concept) {
        if (m_ids.size() <= concept) m_ids.resize(concept + 1);
    }

    ConceptMap& getConceptMap(SnomedConcept concept) {
        reserveConcept(concept);
        return m_ids[concept];
    }
};

class Bucket {
    SnomedId m_bucketStartId;
    SnomedId m_bucketEndId;
    CosineVecValue m_startValue;
    CosineVecValue m_endValue;
    CosineVecValue m_nullValue;
    std::int64_t m_accumulator = 0; // sum of all codes added
    int m_codeCount = 0;            // how many codes were added
    bool m_isDone = false;          // true if addCode was called with a code >= m_bucketEndCode

   public:
    Bucket(SnomedId bucketStartCode,
           SnomedId bucketEndCode,
           CosineVecValue startValue,
           CosineVecValue endValue,
           CosineVecValue nullValue)
        : m_bucketStartId(bucketStartCode),
          m_bucketEndId(bucketEndCode),
          m_startValue(startValue),
          m_endValue(endValue),
          m_nullValue(nullValue) {}

    /**
     * Determines whether the given code is in the bucket's code range
     * @param code the code to test
     * @return true if the code is between the bucket's start (inclusive) and end
     * (exclusive)
     */
    bool isInRange(SnomedId id) const { return id >= m_bucketStartId && !isPastRange(id); }

    /**
     * Returns whether the given code is beyond the high end of the bucket's range
     * @param code the code to test
     * @return true if the given code >= the bucket end code
     */
    bool isPastRange(SnomedId id) const { return id >= m_bucketEndId; }

    /**
     * Returns the distance between the given code and the center of the bucket's
     * range
     * @param code the code to check
     * @return the distance between the code and the bucket's center
     */
    SnomedId distanceFromCenter(SnomedId id) const {
        return SnomedId(abs64(std::int64_t(id) - (m_bucketEndId + m_bucketStartId) / 2));
    }

    /**
     * Attempt to add the given code to the bucket.  If the code is within the
     * bucket's range, accumulate the code
     * into the bucket average.  Otherwise, just ignore the request.
     *
     * @param code the code to add
     */
    void addId(SnomedId id) {
        if (id >= m_bucketEndId)
            m_isDone = true;
        else if (id >= m_bucketStartId) {
            m_accumulator += std::int64_t(id);
            ++m_codeCount;
        }
    }

    /**
     * Returns the average of all added codes, scaled to the active range, or
     * returns the null value if no codes have
     * been added
     *
     * @return the CosineVecValue for all added codes
     */
    CosineVecValue getCosineVecVale() const {
        if (m_codeCount == 0) return m_nullValue;

        double average = double(m_accumulator) / m_codeCount;
        CosineVecValue val = CosineVecValue((average - m_bucketStartId) * (m_endValue - m_startValue) /
                                            (m_bucketEndId - m_bucketStartId)) +
                             m_startValue;
        return val;
    }
};

std::vector<CosineVecValue> makeCosineVector(SnomedConcept concept,
                                             unsigned vectorLength,
                                             const std::vector<SnomedCode>& codes);

} // namespace

#endif /* CODEVECTOR_HPP */
