#ifndef __cBpInsertBuffers_h__
#define __cBpInsertBuffers_h__

#include "common/memdatstruct/cMemoryBlock.h"

using namespace common::memdatstruct;

namespace dstruct {
	namespace paged {
		namespace bptree {

template<class TKey>
class cInsertBuffers
{
public:
	cMemoryBlock* bufferMemBlock;     // memblock for all buffers

	cNodeBuffers<TKey> nodeBuffer;		  // buffers used in node processing

	char *firstItem;
	char *secondItem;
	tNodeIndex* currentPath;
	int* orderCurrentPath;
};

}}}
#endif