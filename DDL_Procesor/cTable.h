#pragma once

#include "dstruct/paged/b+tree/cB+Tree.h"
#include "common/random/cGaussRandomGenerator.h"
#include "common/data/cDataCollection.h"
#include "common/data/cTuplesGenerator.h"
#include "dstruct/paged/core/cBulkLoad.h"
#include "common/datatype/tuple/cCommonNTuple.h"



#include <algorithm>
#include <array>
#include <vector>
#include "cTranslatorCreate.h"
#include "cTranslatorIndex.h"
#include "cCompleteRTree.h"
#include "cCompleteBTree.h"
#include "cRecordGeneratorVar.h"
#include "cRecordGeneratorFix.h"
#include "cCompleteSeqArray.h"
#include "sSystemCatalog.h"



#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>






class cTable
{
public:



	/*proměnné k  vytvoření stromu*/
	TypeOfCreate typeOfTable;
	std::vector<cDataType*> vHeap;//prázný vektor který se přetvoří na haldu jako rekace na create table
	cCompleteSeqArray<cTuple>*seqHeapFix;
	cCompleteSeqArray<cHNTuple>*seqHeapVar;

	bool varlenData;
	bool implicitKeyVarlen;
	bool homogenous;
	unsigned int sizeOfData = sizeof(uint) + sizeof(short);

	cSpaceDescriptor *keySD;//SD pro klič
	cSpaceDescriptor *varlenKeyColumnSD;//SD sloupce který je v klíči
	cDataType *keyType;//datovy typ kliče
	cSpaceDescriptor * SD;//SD pro záznamy v tabulce

	/*univerzalni proměnné*/
	std::vector<cColumn*>*columns;
	string tableName;

	/*generatory*/
	cRecordGeneratorVar *varGen = NULL;
	cRecordGeneratorFix *fixGen = NULL;

	/*Proměnné k indexu typu rtree*/
	std::vector<cCompleteRTree<cTuple>*>*indexesFixLenRTree = NULL;//indexy fixní delky
	std::vector<cCompleteRTree<cHNTuple>*>*indexesVarLenRTree = NULL;//indexy var delky

	/*Proměnné k indexu typu btree*/
	std::vector<cCompleteBTree<cTuple>*>*indexesFixLenBTree = NULL;//indexy fixní delky
	std::vector<cCompleteBTree<cHNTuple>*>*indexesVarLenBTree = NULL;//indexy var delky


	/*clustered table*/
	cSpaceDescriptor *clusteredDataSD = NULL;

	
	

public:

	cTable();
	bool CreateTable(string query, cQuickDB *quickDB, const unsigned int blockSize, const unsigned int cacheSize, uint DSMODE, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize);
	bool CreateIndex(string query, cQuickDB *quickDB, const unsigned int blockSize, uint DSMODE, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize);
	bool CreateClusteredTable(cTranslatorCreate *translator, cQuickDB *quickDB, const unsigned int BLOCK_SIZE, uint DSMODE, static const uint structure, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize);

	void SetValues(cTuple *tuple, cSpaceDescriptor *SD);
	void SetValues(cHNTuple *tuple, cSpaceDescriptor *SD);

	
	cTuple * TransportItemFixLen(cTuple *sourceTuple, cSpaceDescriptor *mSd, int columnPosition, cDataType *mType);
	cTuple * TransportItemFixLen(cHNTuple *sourceTuple, cSpaceDescriptor *mSd, int columnPosition, cDataType *mType);
	cHNTuple * TransportItemVarLen(cTuple *sourceTuple, cSpaceDescriptor *columnSD, cSpaceDescriptor *keySD, int columnPosition, cDataType *mType);
	cTuple * TransportItemVarLen(cHNTuple *sourceTuple, cSpaceDescriptor *columnSD, cSpaceDescriptor *keySD, int columnPosition, cDataType *mType);
	cTuple *TransportItemHomoVarlen(cTuple *sourceTuple, cSpaceDescriptor *columnSD, cSpaceDescriptor *keySD, int columnPosition, cDataType *mType);
	
	char *ExtractDataFixLen(cTuple *sourceTuple, int keyPosition,cSpaceDescriptor *dataSD);
	char *ExtractDataFixLen(cHNTuple *sourceTuple, int keyPosition, cSpaceDescriptor *dataSD);
	char *ExtractDataHomoVarlen(cTuple *sourceTuple, int keyPosition, cSpaceDescriptor *dataSD);
	char *ExtractVarData(cHNTuple *sourceTuple, int keyPosition, cSpaceDescriptor *dataSD);

	bool ConstructIndexBtree(string indexName, cDataType *indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor *indexSD, cSpaceDescriptor *indexKeyColumnSD,bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize, cQuickDB *quickDB);
	bool ConstructIndexRtree(string indexName, cDataType *indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor *indexSD, cSpaceDescriptor *indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, cQuickDB *quickDB);
	
	
	bool ConstructVarlenHomoIndexBTree(string indexName, cDataType * indexType, cSpaceDescriptor *indexKeyColumnSD, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, const uint inMemCacheSize, cQuickDB * quickDB);
	bool ConstructVarlenIndexBTree(string indexName, cDataType * indexType, cSpaceDescriptor *indexKeyColumnSD, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, const uint inMemCacheSize, cQuickDB * quickDB);
	bool ConstructVarlenHomoIndexRTree(string indexName, cDataType * indexType, cSpaceDescriptor *indexKeyColumnSD, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, uint dsMode,  uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms,  cQuickDB * quickDB);
	
	
	char* LoadIndexData(unsigned int nodeId, unsigned int position);

	/*nové staré metody  7.4.*/
	cHNTuple* TransportItemVarLenCHNTuple(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType);//7.4
	cHNTuple* TransportItemHomoVarlenCHNTuple(cTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType);//7.4
	cHNTuple *TransportItemFixLenCHNTuple(cTuple *sourceTuple, cSpaceDescriptor *keySD, int columnPosition, cDataType *mType);//7.4
	cHNTuple* TransportItemFixLenCHNTuple(cHNTuple * sourceTuple, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType);//7.4
	bool ConstructIndexBtreeVar(string indexName, cDataType *indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor *indexSD, cSpaceDescriptor *indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize, cQuickDB *quickDB);
	bool ConstructIndexRtreeVar(string indexName, cDataType *indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor *indexSD, cSpaceDescriptor *indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, cQuickDB *quickDB);
	
	cTreeItemStream<cTuple>* Find(int key);
	bool Serialization();
};

cTable::cTable():vHeap(NULL),varlenKeyColumnSD(NULL)
{
	
}

inline bool cTable::CreateTable(string query, cQuickDB *quickDB, const unsigned int blockSize, const unsigned int cacheSize, uint DSMODE, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize)
{
	//bstrom1
	cTranslatorCreate *translator = new cTranslatorCreate();//instance překladače



	translator->TranlateCreate(query);//překladad cretae table

	keySD = translator->keySD;
	keyType = translator->keyDimensionType;
	SD = translator->SD;
	columns = translator->columns;
	tableName = translator->tableName;
	varlenData = translator->varlenData;
	implicitKeyVarlen = translator->keyVarlen;
	typeOfTable = translator->typeOfCreate;
	homogenous = translator->homogenous;

	for (int i = 0; i < columns->size(); i++)
	{
		if (columns->at(i)->primaryKey)
		{
			varlenKeyColumnSD = columns->at(i)->columnSD;
		}

	}
	
	Serialization();

	if (((implicitKeyVarlen && varlenData) && homogenous==false) || (implicitKeyVarlen &&  homogenous == false) || (implicitKeyVarlen == false && homogenous == false)&& varlenData)
	{
		varGen = new cRecordGeneratorVar(columns, SD);
			
		if (typeOfTable != CLUSTERED_TABLE_BTREE && typeOfTable != CLUSTERED_TABLE_RTREE)
		{
			seqHeapVar = new cCompleteSeqArray<cHNTuple>((tableName + "_heap").c_str(), blockSize, cacheSize, SD, quickDB);
		}
	}
	else
	{
		fixGen = new cRecordGeneratorFix(columns, SD);
		if (typeOfTable != CLUSTERED_TABLE_BTREE && typeOfTable != CLUSTERED_TABLE_RTREE)
		{ 
			seqHeapFix = new cCompleteSeqArray<cTuple>((tableName + "_heap").c_str(), blockSize, cacheSize, SD, quickDB);
		}
		
	}	

	
	
	if (typeOfTable == BTREE)
	{
		indexesFixLenBTree = new vector<cCompleteBTree<cTuple>*>();
		indexesVarLenBTree = new vector<cCompleteBTree<cHNTuple>*>();




		unsigned int indexSD = keySD->GetSize();
		//unsigned int lengthIndexSD = tp->GetLength();//pocet sloupcu
		unsigned int typeSizeIndex = keySD->GetTypeSize();


		unsigned int lengthkeySD = sizeof(int);
		//unsigned int typeSizeKey = varlenKeyColumnSD->GetTypeSize();


		

		if (implicitKeyVarlen)
		{
			cCompleteBTree<cHNTuple>*index = new cCompleteBTree<cHNTuple>(tableName.c_str(), translator->keyPosition, blockSize, keySD, keySD->GetTypeSize(), sizeOfData, false, DSMODE, cDStructConst::BTREE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);
			if (index != NULL)
			{
				indexesVarLenBTree->push_back(index);
				return true;
			}
		}
		else
		{
			cCompleteBTree<cTuple>*index = new cCompleteBTree<cTuple>(tableName.c_str(), translator->keyPosition, blockSize, keySD, keySD->GetTypeSize(), sizeOfData, false, DSMODE, cDStructConst::BTREE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);
			if (index != NULL)
			{
				indexesFixLenBTree->push_back(index);
				return true;
			}
		}

		

	}
	else if (typeOfTable == RTREE)
	{
		indexesFixLenRTree = new vector<cCompleteRTree<cTuple>*>();
		indexesVarLenRTree = new vector<cCompleteRTree<cHNTuple>*>();
		
		if (varlenData || implicitKeyVarlen)
		{
			cout << "varlen data or varlen key not suported in R-tree" << endl;
			cin.get();
			exit(0);
		}
		else
		{


			cCompleteRTree<cTuple>*index = new cCompleteRTree<cTuple>(tableName.c_str(), translator->keyPosition, blockSize, keySD, keySD->GetTypeSize(), sizeOfData, false, DSMODE, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);


			if (index != NULL)
			{
				indexesFixLenRTree->push_back(index);

				return true;
			}
		}
		
		/*if (implicitKeyVarlen == false)
		{
			cCompleteRTree<cTuple>*index = new cCompleteRTree<cTuple>(tableName.c_str(), translator->keyPosition, blockSize, keySD, keySD->GetTypeSize(), sizeOfData, false, DSMODE, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);


			if (index != NULL)
			{
				indexesFixLenRTree->push_back(index);

				return true;
			}
		}
		else
		{
			unsigned int gu = keySD->GetDimensionSize(0);

			cCompleteRTree<cHNTuple>*index = new cCompleteRTree<cHNTuple>(tableName.c_str(), translator->keyPosition, blockSize, keySD, keySD->GetTypeSize(), sizeOfData, false, DSMODE, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);
			if (index != NULL)
			{
				indexesVarLenRTree->push_back(index);
				return true;
			}
		}*/
		

	}
	else if(typeOfTable ==CLUSTERED_TABLE_BTREE)
	{
		CreateClusteredTable(translator, quickDB, blockSize, DSMODE, cDStructConst::RTREE  ,compressionRatio,  codeType,  runtimeMode,  histograms, inMemCacheSize);
	}
	else if (typeOfTable == CLUSTERED_TABLE_BTREE)
	{
		CreateClusteredTable(translator, quickDB, blockSize, DSMODE, cDStructConst::BTREE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize);
	}
	else
	{
		cout << "unknow type of table" << endl;
		exit(0);
	}

	
}

inline bool cTable::CreateIndex(string query, cQuickDB * quickDB, const unsigned int BLOCK_SIZE, uint DSMODE, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, const uint inMemCacheSize)
{


	cTranslatorIndex *translator = new cTranslatorIndex();//instance překladače
	translator->TranslateCreateIndex(query);



	cDataType *indexType=NULL;
	int size;
	int indexColumnPosition;
	string indexName = translator->indexName;
	bool varlenIndex=false;

	

	cSpaceDescriptor *indexKeyColumnSD=NULL;//SD klíčového sloupce


	if (translator->tableName.compare(tableName) == 0)//porovnání jména tabulky
	{
		bool found=false;
		for (int i = 0; i < columns->size(); i++)
		{
			string name = columns->at(i)->name;
			if (name.compare(translator->columnName) == 0)//pokud sloupec existuje tak..
			{
				found = true;
				indexColumnPosition = columns->at(i)->positionInTable;//pozice v tabulce
				indexType = columns->at(i)->cType;//typ
				size = columns->at(i)->size;
				if (size != 0)
				{
					varlenIndex = true;//yji3t2n9 jestli je slopec fixlen nebo varlenData
					indexKeyColumnSD = columns->at(i)->columnSD;
				}
				else
					indexKeyColumnSD = NULL;
				i = columns->size();//vyzkočení z foru
			}
		}
		if (found == false)
		{
			cout << "column dont exist" << endl;
			cin.get();
			exit(0);
		}
	}
	else
	{
		cout << "table dont exist" << endl;
		cin.get();
		exit(0);
	}

	cSpaceDescriptor *indexSD = NULL;//SD indexu
	if (varlenIndex)
	{
		indexSD = new cSpaceDescriptor(1, new cHNTuple(), indexType, false);
		indexSD->SetDimSpaceDescriptor(0, indexKeyColumnSD);
		indexSD->SetDimensionType(0, indexType);

		indexSD->Setup();
	}
	else
	{
		indexSD = new cSpaceDescriptor(1, new cTuple(), indexType, false);
	}

	if (typeOfTable == RTREE)
	{
		
		if (varlenIndex==false)
		{
			ConstructIndexRtree(indexName.c_str(), indexType, indexColumnPosition, BLOCK_SIZE, indexSD, indexKeyColumnSD, varlenIndex, DSMODE, compressionRatio, codeType, runtimeMode, histograms, quickDB);
		}
		else
		{
			ConstructIndexRtreeVar(indexName.c_str(), indexType, indexColumnPosition, BLOCK_SIZE, indexSD, indexKeyColumnSD, varlenIndex, DSMODE, compressionRatio, codeType, runtimeMode, histograms, quickDB);
		}
			
		
	}
	else
	{
		if (varlenIndex == false)
		{
			ConstructIndexBtree(indexName.c_str(), indexType, indexColumnPosition, BLOCK_SIZE, indexSD, indexKeyColumnSD, varlenIndex, DSMODE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);
		}
		else
		{
			ConstructIndexBtreeVar(indexName.c_str(), indexType, indexColumnPosition, BLOCK_SIZE, indexSD, indexKeyColumnSD, varlenIndex, DSMODE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);
		}
	}
	}

inline bool cTable::CreateClusteredTable(cTranslatorCreate *translator, cQuickDB *quickDB, const unsigned int BLOCK_SIZE, uint DSMODE, static const uint structure, unsigned int compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize)
{

	/*****************************************************/
	/*zjištění, kolik sloupců s SD je v záznamu*/
	cSpaceDescriptor ** ptrSD;
	int numberOfColumnSDs = 0;
	for (int i = 0; i < columns->size(); i++)
	{
		if (columns->at(i)->primaryKey == NULL)
		{
			if (columns->at(i)->columnSD != NULL)
			{
				numberOfColumnSDs++;
			}
		}
	}
	/*vložení sd do pole*/
	ptrSD = new cSpaceDescriptor*[numberOfColumnSDs];
	for (int i = 0, k = 0; i < columns->size(); i++)
	{
		if (columns->at(i)->primaryKey == NULL)
		{
			if (columns->at(i)->columnSD != NULL)
			{
				ptrSD[k] = columns->at(i)->columnSD;
				k++;
			}
		}
	}
	/***********************************************/



	/*vytvoření sd pro data */
	cDataType ** typeArray;
	int sizeOfSD = columns->size() - 1;
	typeArray = new cDataType*[sizeOfSD];//-klíč
	for (int i = 0, j = 0; i < columns->size(); i++)
	{

		if (columns->at(i)->primaryKey == NULL)
		{
			typeArray[j] = columns->at(i)->cType;
			j++;
		}
	}

	if (varlenData)
	{
		clusteredDataSD = new cSpaceDescriptor(sizeOfSD, new cHNTuple(), typeArray, false);
	}
	else
	{
		clusteredDataSD = new cSpaceDescriptor(sizeOfSD, new cTuple(), typeArray, false);
	}

	/*nastavení dim sd*/
	for (int i = 0, k = 0; i < sizeOfSD; i++)
	{
		if (typeArray[i]->GetCode() == 'n')
		{
			clusteredDataSD->SetDimSpaceDescriptor(i, ptrSD[k]);
			clusteredDataSD->SetDimensionType(i, typeArray[i]);
		}
	}


	clusteredDataSD->Setup();







	if (typeOfTable == CLUSTERED_TABLE_BTREE)
	{
		indexesFixLenBTree = new vector<cCompleteBTree<cTuple>*>();
		indexesVarLenBTree = new vector<cCompleteBTree<cHNTuple>*>();


		if (implicitKeyVarlen)
		{


			unsigned int keySize = keySD->GetSize();
			unsigned int pom = keySD->GetDimensionSize(0);
			unsigned int pom2 = SD->GetSize() - keySize;
			unsigned int pom3 = clusteredDataSD->GetSize();

			cCompleteBTree<cHNTuple>*index = new cCompleteBTree<cHNTuple>(tableName.c_str(), translator->keyPosition, BLOCK_SIZE, keySD, keySD->GetTypeSize(), clusteredDataSD->GetTypeSize(), varlenData, DSMODE, structure, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);


			if (index != NULL)
			{
				indexesVarLenBTree->push_back(index);
				//indexesFixLen->push_back(index);
				return true;
			}

		}
		else if (implicitKeyVarlen == false)
		{

			unsigned int keySize = keySD->GetSize();

			unsigned int pom2 = SD->GetSize() - keySize;
			unsigned int pom3 = clusteredDataSD->GetTypeSize();
			unsigned int pom4 = clusteredDataSD->GetDimensionSize(0);

			cCompleteBTree<cTuple>*index = new cCompleteBTree<cTuple>(tableName.c_str(), translator->keyPosition, BLOCK_SIZE, keySD, keySD->GetTypeSize(), clusteredDataSD->GetTypeSize(), varlenData, DSMODE, structure, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);


			if (index != NULL)
			{
				indexesFixLenBTree->push_back(index);
				//indexesFixLen->push_back(index);
				return true;
			}
		}
		else
		{
			return false;
		}
	}
	else if (typeOfTable == CLUSTERED_TABLE_RTREE)
	{
		indexesFixLenRTree = new vector<cCompleteRTree<cTuple>*>();
		indexesVarLenRTree = new vector<cCompleteRTree<cHNTuple>*>();


		if (implicitKeyVarlen || varlenData)
		{
			cout << "varlen data or varlen key not suported in R-tree" << endl;
			cin.get();
			exit(0);

			

		}
		else
		{

			unsigned int keySize = keySD->GetSize();

			unsigned int pom2 = SD->GetSize() - keySize;
			unsigned int pom3 = clusteredDataSD->GetTypeSize();
			unsigned int pom4 = clusteredDataSD->GetDimensionSize(0);

			cCompleteRTree<cTuple>*index = new cCompleteRTree<cTuple>(tableName.c_str(), translator->keyPosition, BLOCK_SIZE, keySD, keySD->GetTypeSize(), clusteredDataSD->GetTypeSize(), varlenData, DSMODE, structure, compressionRatio, codeType, runtimeMode, histograms, quickDB);


			if (index != NULL)
			{
				indexesFixLenRTree->push_back(index);
				//indexesFixLen->push_back(index);
				return true;
			}
		}
		
	}
}



inline void cTable::SetValues(cTuple * tuple, cSpaceDescriptor * SD)
	{

		if (typeOfTable == RTREE)
		{

			if (implicitKeyVarlen == false)
			{
				cRTree<cTuple> *mKeyIndex = indexesFixLenRTree->at(0)->mIndex;
				cTuple *keyTuple = NULL;
				/*if (homogenous)
				{*/
					
					keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);
		
				//}

				unsigned int nodeid, position;
				seqHeapFix->mSeqArray->AddItem(nodeid, position, *tuple);
				mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			}
			else
			{
				cRTree<cHNTuple> *mKeyIndex = indexesVarLenRTree->at(0)->mIndex;
				cHNTuple *keyTuple = NULL;
				if (homogenous)
				{

					keyTuple = TransportItemHomoVarlenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenRTree->at(0)->indexColumnPosition, keyType);
				}
				else
				{
					//keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);//asi nikdy nenastane tato možnost
					cout << "wtf" << endl;
				}

				unsigned int nodeid, position;
				seqHeapFix->mSeqArray->AddItem(nodeid, position, *tuple);
				mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			}

		}
		else if (typeOfTable == BTREE)
		{

			if (implicitKeyVarlen == false)
			{
				cBpTree<cTuple> *mKeyIndex = indexesFixLenBTree->at(0)->mIndex;
				cTuple *keyTuple = NULL;
				/*if (homogenous)
				{*/
					keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenBTree->at(0)->indexColumnPosition, keyType);

				//}

				unsigned int nodeid, position;
				seqHeapFix->mSeqArray->AddItem(nodeid, position, *tuple);
				mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			}
			else
			{
				cBpTree<cHNTuple> *mKeyIndex = indexesVarLenBTree->at(0)->mIndex;
				cHNTuple *keyTuple = NULL;
				if (homogenous)
				{

					keyTuple = TransportItemHomoVarlenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenBTree->at(0)->indexColumnPosition, keyType);
				}
				else
				{
					//keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);//asi nikdy nenastane tato možnost
					cout << "wtf" << endl;
				}

				unsigned int nodeid, position;
				seqHeapFix->mSeqArray->AddItem(nodeid, position, *tuple);
				mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			}

		}
		else if(typeOfTable==CLUSTERED_TABLE_BTREE)
		{
			if (implicitKeyVarlen == false)
			{
				cBpTree<cTuple> *mKeyIndex = indexesFixLenBTree->at(0)->mIndex;
				cTuple *keyTuple = NULL;

				if (homogenous)
				{
					keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenBTree->at(0)->indexColumnPosition, keyType);

				}

				//Nahrani data
				char *clusteredData;

				if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
				{

					
					clusteredData = ExtractDataFixLen(tuple, indexesFixLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
					if (clusteredData == NULL)
					{
						cout << "hm?" << endl;
					}
				}
				else
				{
					clusteredData = ExtractDataHomoVarlen(tuple, indexesFixLenBTree->at(0)->indexColumnPosition, clusteredDataSD);
				}
				//
				mKeyIndex->Insert(*keyTuple, clusteredData);
			}
			else
			{
				cBpTree<cHNTuple> *mKeyIndex = indexesVarLenBTree->at(0)->mIndex;
				cHNTuple *keyTuple = NULL;
				if (homogenous)
				{

					keyTuple = TransportItemHomoVarlenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenBTree->at(0)->indexColumnPosition, keyType);//jde
				}
				else
				{
					//keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);//asi nikdy nenastane tato možnost
					cout << "wtf" << endl;
				}
				//Nahrani data
				char *clusteredData;

				if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
				{
					clusteredData = ExtractDataFixLen(tuple, indexesFixLenBTree->at(0)->indexColumnPosition, clusteredDataSD);
				}
				else
				{
					clusteredData = ExtractDataHomoVarlen(tuple, indexesVarLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
				}

				mKeyIndex->Insert(*keyTuple, clusteredData);
			
			}
			


		}
		else if (typeOfTable == CLUSTERED_TABLE_RTREE)
		{
			cRTree<cTuple> *mKeyIndex = indexesFixLenRTree->at(0)->mIndex;
			if (implicitKeyVarlen == false)
			{
				cRTree<cTuple> *mKeyIndex = indexesFixLenRTree->at(0)->mIndex;
				cTuple *keyTuple = NULL;

				if (homogenous)
				{
					keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);

				}

				//Nahrani data
				char *clusteredData;

				if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
				{


					clusteredData = ExtractDataFixLen(tuple, indexesFixLenRTree->at(0)->indexColumnPosition, clusteredDataSD);//mrdá všechno;sd je v pořdáku,
					if (clusteredData == NULL)
					{
						cout << "hm?" << endl;
					}
				}
				else
				{
					clusteredData = ExtractDataHomoVarlen(tuple, indexesFixLenRTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
				}
				//
				mKeyIndex->Insert(*keyTuple, clusteredData);
			}
			else
			{
				cRTree<cHNTuple> *mKeyIndex = indexesVarLenRTree->at(0)->mIndex;
				cHNTuple *keyTuple = NULL;
				if (homogenous)
				{

					keyTuple = TransportItemHomoVarlenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenRTree->at(0)->indexColumnPosition, keyType);//jde
				}
				else
				{
					//keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);//asi nikdy nenastane tato možnost
					cout << "wtf" << endl;
				}
				//Nahrani data
				char *clusteredData;

				if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
				{
					clusteredData = ExtractDataFixLen(tuple, indexesFixLenRTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
				}
				else
				{
					clusteredData = ExtractDataHomoVarlen(tuple, indexesVarLenRTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
				}

				mKeyIndex->Insert(*keyTuple, clusteredData);
			}
		}
		else
		{
			cout << "Error in set tuple method" << endl;
			cin.get();
			exit(0);
		}
}

inline void cTable::SetValues(cHNTuple *tuple, cSpaceDescriptor *SD)//nastavení hodnopty záznamu a vložení primárního klíče do b-stromu pro primarni kliče
{
	
	if (typeOfTable == RTREE)
	{

		if (implicitKeyVarlen == false)
		{
			cRTree<cTuple> *mKeyIndex = indexesFixLenRTree->at(0)->mIndex;
			cTuple *keyTuple = NULL;

			
			keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);
			

			int key = keyTuple->GetInt(0, keySD);

			unsigned int nodeid, position;
			seqHeapVar->mSeqArray->AddItem(nodeid, position, *tuple);
			bool hm = mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			if (hm)
			{
				cout << "value inserted" << endl;
			}
			else
			{
				cout << "not inserted" << endl;
			}

		}
		else
		{
			cRTree<cHNTuple> *mKeyIndex = indexesVarLenRTree->at(0)->mIndex;
			cHNTuple *keyTuple = NULL;

			
			keyTuple = TransportItemVarLenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenRTree->at(0)->indexColumnPosition, keyType);
			

			int key = keyTuple->GetInt(0, keySD);

			unsigned int nodeid, position;
			seqHeapVar->mSeqArray->AddItem(nodeid, position, *tuple);
			bool hm = mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			if (hm)
			{
				cout << "value inserted" << endl;
			}
			else
			{
				cout << "not inserted" << endl;
			}
		}

		
	}
	else if(typeOfTable==BTREE)
	{
		
		if (implicitKeyVarlen == false)
		{
			cBpTree<cTuple> *mKeyIndex = indexesFixLenBTree->at(0)->mIndex;
			cTuple *keyTuple = NULL;


			keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenBTree->at(0)->indexColumnPosition, keyType);


			int key = keyTuple->GetInt(0, keySD);

			unsigned int nodeid, position;
			seqHeapVar->mSeqArray->AddItem(nodeid, position, *tuple);
			bool hm = mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			if (hm)
			{
				cout << "value inserted" << endl;
			}
			else
			{
				cout << "not inserted" << endl;
			}

		}
		else
		{
			cBpTree<cHNTuple> *mKeyIndex = indexesVarLenBTree->at(0)->mIndex;
			cHNTuple *keyTuple = NULL;


			keyTuple = TransportItemVarLenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenBTree->at(0)->indexColumnPosition, keyType);


			int key = keyTuple->GetInt(0, keySD);

			unsigned int nodeid, position;
			seqHeapVar->mSeqArray->AddItem(nodeid, position, *tuple);
			bool hm = mKeyIndex->Insert(*keyTuple, LoadIndexData(nodeid, position));
			if (hm)
			{
				cout << "value inserted" << endl;
			}
			else
			{
				cout << "not inserted" << endl;
			}
		}
			
	}
	else if (typeOfTable == CLUSTERED_TABLE_BTREE)
	{
		if (implicitKeyVarlen == false)
		{
			cBpTree<cTuple> *mKeyIndex = indexesFixLenBTree->at(0)->mIndex;
			cTuple *keyTuple = NULL;


			keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenBTree->at(0)->indexColumnPosition, keyType);


			//Nahrani data
			char *clusteredData;

			if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
			{
				clusteredData = ExtractDataFixLen(tuple, indexesFixLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//nedostal jsem se tu
			}
			else
			{
				clusteredData = ExtractVarData(tuple, indexesFixLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//nejde zařve assert
			}

			mKeyIndex->Insert(*keyTuple, clusteredData);

		}
		else
		{
			cBpTree<cHNTuple> *mKeyIndex = indexesVarLenBTree->at(0)->mIndex;
			cHNTuple *keyTuple = NULL;


			keyTuple = TransportItemVarLenCHNTuple(tuple, varlenKeyColumnSD, keySD, indexesVarLenBTree->at(0)->indexColumnPosition, keyType);


			//Nahrani data
			char *clusteredData;

			if ((implicitKeyVarlen && varlenData == false) || (homogenous && implicitKeyVarlen == false))
			{
				clusteredData = ExtractDataFixLen(tuple, indexesVarLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//jde
			}
			else
			{
				clusteredData = ExtractVarData(tuple, indexesVarLenBTree->at(0)->indexColumnPosition, clusteredDataSD);//nejde zařve assert
			}

			mKeyIndex->Insert(*keyTuple, clusteredData);
		}

		


	}
	else if (typeOfTable == CLUSTERED_TABLE_RTREE)
	{
		cRTree<cTuple> *mKeyIndex = indexesFixLenRTree->at(0)->mIndex;
		cTuple *keyTuple = new cTuple();
		if (implicitKeyVarlen == false)
		{
			keyTuple = TransportItemFixLen(tuple, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);
		}
		else
		{
			keyTuple = TransportItemVarLen(tuple, varlenKeyColumnSD, keySD, indexesFixLenRTree->at(0)->indexColumnPosition, keyType);
		}
		int h = keyTuple->GetInt(0, keySD);

		mKeyIndex->Insert(*keyTuple, NULL);
	}
	else
	{
		cout << "Error in set tuple method" << endl;
		cin.get();
		exit(0);
	}
}






inline cTuple * cTable::TransportItemFixLen(cTuple *sourceTuple, cSpaceDescriptor *mSd, int columnPosition, cDataType *mType)
{
	cTuple *destTuple=new cTuple(mSd);
	
	if (mType->GetCode() == 'i')//int
	{
		int key = sourceTuple->GetInt(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'f')//float, nepodporovan
	{
		float key = sourceTuple->GetFloat(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 's')//short
	{
		short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'S')//unsigned short
	{
		unsigned short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'u')//long
	{

		uint key = sourceTuple->GetUInt(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	return destTuple;
}
inline cHNTuple * cTable::TransportItemFixLenCHNTuple(cTuple *sourceTuple, cSpaceDescriptor *keySD, int columnPosition, cDataType *mType)
{
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'i')//int
	{
		int key = sourceTuple->GetInt(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'f')//float, nepodporovan
	{
		float key = sourceTuple->GetFloat(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 's')//short
	{
		short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'S')//unsigned short
	{
		unsigned short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'u')//long
	{

		uint key = sourceTuple->GetUInt(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	return destTuple;
}

inline cTuple * cTable::TransportItemFixLen(cHNTuple * sourceTuple, cSpaceDescriptor * mSd,  int columnPosition, cDataType * mType)
{
	cTuple *destTuple = new cTuple(mSd);


	if (mType->GetCode() == 'i')//int
	{
		int key = sourceTuple->GetInt(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'f')//float, nepodporovan
	{
		float key = sourceTuple->GetFloat(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 's')//short
	{
		short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'S')//unsigned short
	{
		unsigned short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	else if (mType->GetCode() == 'u')//long
	{

		uint key = sourceTuple->GetUInt(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}
	return destTuple;
}
inline cHNTuple * cTable::TransportItemFixLenCHNTuple(cHNTuple * sourceTuple, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
{
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'i')//int
	{
		int key = sourceTuple->GetInt(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'f')//float, nepodporovan?
	{
		float key = sourceTuple->GetFloat(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 's')//short
	{
		short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'S')//unsigned short
	{
		unsigned short key = sourceTuple->GetUShort(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	else if (mType->GetCode() == 'u')//long
	{

		uint key = sourceTuple->GetUInt(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}
	return destTuple;
}





inline cHNTuple * cTable::TransportItemVarLen(cTuple * sourceTuple, cSpaceDescriptor * mSd, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
{
	cNTuple *varlenTuple = new cNTuple(mSd);
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'n')//varchar,neodzkoušeno
	{
		
		char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), columnPosition, SD);
		varlenTuple->SetData(TEMPTuple);
		destTuple->SetValue(0, *varlenTuple, keySD);


	}
	else if (mType->GetCode() == 'c')//char(nejasny Get)
	{
		char key = sourceTuple->GetWChar(columnPosition, SD);
		destTuple->SetValue(0, key, mSd);
	}

	return destTuple;
}

inline cTuple * cTable::TransportItemVarLen(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
{

		cNTuple *varlenTuple = new cNTuple(columnSD);
		cTuple *destTuple = new cTuple(keySD);
	
		
		if (mType->GetCode() == 'n')
		{

			char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), columnPosition, SD);
			varlenTuple->SetData(TEMPTuple);

			char a = varlenTuple->GetByte(0, columnSD);
			
			destTuple->SetValue(0, *varlenTuple, keySD);
		}
		else if (mType->GetCode() == 'c')//char(nejasny Get)
		{
			char key = sourceTuple->GetWChar(columnPosition, SD);
			destTuple->SetValue(0, key, keySD);
		}
	
		return destTuple;
}
inline cHNTuple * cTable::TransportItemVarLenCHNTuple(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
{

	cNTuple *varlenTuple = new cNTuple(columnSD);
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'n')
	{

		char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), columnPosition, SD);
		varlenTuple->SetData(TEMPTuple);

		char a = varlenTuple->GetByte(0, columnSD);

		destTuple->SetValue(0, *varlenTuple, keySD);
	}
	else if (mType->GetCode() == 'c')//char(nejasny Get)
	{
		char key = sourceTuple->GetWChar(columnPosition, SD);
		destTuple->SetValue(0, key, keySD);
	}

	return destTuple;
}



inline cTuple * cTable::TransportItemHomoVarlen(cTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)//asi nikdy nenastane?
{
	cNTuple *varlenTuple = new cNTuple(columnSD);
	cTuple *destTuple = new cTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'n')//varchar,neodzkoušeno
	{
		char* huihua = sourceTuple->GetData();
		char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), columnPosition, SD);
		varlenTuple->SetData(TEMPTuple);
		destTuple->SetValue(0, *varlenTuple, keySD);

		///varcharTuple->SetData(TEMPTuple);
		//destTuple->SetValue(0, *varcharTuple, SD);
		//char a=destTuple->GetCChar(0, mSd);

	}
	else if (mType->GetCode() == 'c')//char(nejasny Get)
	{
		char key = sourceTuple->GetWChar(columnPosition, SD);
		destTuple->SetValue(0, key, columnSD);
	}

	return destTuple;
}
inline cHNTuple * cTable::TransportItemHomoVarlenCHNTuple(cTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
{
	cNTuple *varlenTuple = new cNTuple(columnSD);
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(keySD);

	if (mType->GetCode() == 'n')
	{
		char* huihua = sourceTuple->GetData();
		char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), columnPosition, SD);
		varlenTuple->SetData(TEMPTuple);
		destTuple->SetValue(0, *varlenTuple, keySD);

		///varcharTuple->SetData(TEMPTuple);
		//destTuple->SetValue(0, *varcharTuple, SD);
		//char a=destTuple->GetCChar(0, mSd);

	}
	else if (mType->GetCode() == 'c')//char(nejasny Get)
	{
		char key = sourceTuple->GetWChar(columnPosition, SD);
		destTuple->SetValue(0, key, columnSD);
	}

	return destTuple;
}

inline char * cTable::ExtractVarData(cHNTuple * sourceTuple, int keyPosition, cSpaceDescriptor * dataSD)
{


	char *extractedData;


	cNTuple *varlenTuple = NULL;
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(dataSD);

	for (int i = 0,j=0; i < SD->GetDimension(); i++)
	{
		if (i != keyPosition)
		{
			cDataType*mType = dataSD->GetDimensionType(j);
			if (mType->GetCode() == 'i')//int
			{
				int item = sourceTuple->GetInt(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'n')
			{
				char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), i, SD);
				varlenTuple = new cNTuple(SD->GetDimSpaceDescriptor(i));
				varlenTuple->SetData(TEMPTuple);
				destTuple->SetValue(j, *varlenTuple, dataSD);
				j++;

			}
			else if (mType->GetCode() == 'f')//float, nepodporovan
			{
				float item = sourceTuple->GetFloat(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'c')//char(nejasny Get)
			{
				char item = sourceTuple->GetWChar(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 's')//short
			{
				short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'S')//unsigned short
			{
				unsigned short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'u')//long
			{

				uint item = sourceTuple->GetUInt(i, SD);
				destTuple->SetValue(0, item, dataSD);
				j++;
			}

		}
	}
	extractedData = destTuple->GetData();

	return extractedData;
}

inline char * cTable::ExtractDataFixLen(cTuple * sourceTuple, int keyPosition, cSpaceDescriptor * dataSD)
{
	cTuple *destTuple = new cTuple(dataSD);


	
	char *extractedData;
	
	for (int i = 0, j = 0; i < SD->GetDimension(); i++)
	{
		if (i != keyPosition)
		{
			cDataType*mType = dataSD->GetDimensionType(j);
			if (mType->GetCode() == 'i')//int
			{
				int item = sourceTuple->GetInt(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'u')//uint
			{
				unsigned int item = sourceTuple->GetUInt(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'f')//float, nepodporovan
			{
				float item = sourceTuple->GetFloat(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 's')//short
			{
				short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'S')//unsigned short
			{
				unsigned short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'u')//long
			{
				uint item = sourceTuple->GetUInt(i, SD);
				destTuple->SetValue(0, item, dataSD);
				j++;
			}
		}
	}
	


	return extractedData = destTuple->GetData();;


}

inline char * cTable::ExtractDataFixLen(cHNTuple * sourceTuple, int keyPosition, cSpaceDescriptor * dataSD)
{
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(dataSD);


	char *extractedData;

	for (int i = 0, j = 0; i < SD->GetDimension(); i++)
	{
		if (i != keyPosition)
		{
			cDataType*mType = dataSD->GetDimensionType(j);
			if (mType->GetCode() == 'i')//int
			{
				int item = sourceTuple->GetInt(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'u')//uint
			{
				unsigned int item = sourceTuple->GetUInt(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'f')//float, nepodporovan
			{
				float item = sourceTuple->GetFloat(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 's')//short
			{
				short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'S')//unsigned short
			{
				unsigned short item = sourceTuple->GetUShort(i, SD);
				destTuple->SetValue(j, item, dataSD);
				j++;
			}
			else if (mType->GetCode() == 'u')//long
			{

				uint item = sourceTuple->GetUInt(i, SD);
				destTuple->SetValue(0, item, dataSD);
				j++;
			}
		}
	}
	return extractedData=destTuple->GetData();
}

inline char * cTable::ExtractDataHomoVarlen(cTuple * sourceTuple, int keyPosition, cSpaceDescriptor * dataSD)
{
	char *extractedData;


	cNTuple *varlenTuple = NULL;
	cHNTuple *destTuple = new cHNTuple();
	destTuple->Resize(dataSD);


	for (int i = 0, j = 0; i < SD->GetDimension(); i++)
	{
		if (i != keyPosition)
		{
			cDataType*mType = dataSD->GetDimensionType(j);
			if (mType->GetCode() == 'n')
			{
				char * TEMPTuple = sourceTuple->GetTuple(sourceTuple->GetData(), i, SD);
				varlenTuple = new cNTuple(SD->GetDimSpaceDescriptor(i));
				varlenTuple->SetData(TEMPTuple);
				destTuple->SetValue(j, *varlenTuple, dataSD);
				j++;
			}

		}
	}
	extractedData = destTuple->GetData();
	return extractedData;
}

inline bool cTable::ConstructIndexBtree(string indexName,cDataType *indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor *indexSD, cSpaceDescriptor *indexKeyColumnSD,bool varlenKey, uint dsMode,  uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, static const uint inMemCacheSize, cQuickDB *quickDB)
{
	cCompleteBTree<cTuple> *index = new cCompleteBTree<cTuple>(indexName.c_str(), indexColumnPosition, blockSize, indexSD, indexSD->GetTypeSize(), sizeOfData/*indexSD->GetSize()*/, false, dsMode, cDStructConst::BTREE, compressionRatio, codeType, runtimeMode, histograms, inMemCacheSize, quickDB);

	//cCompleteRTree<cTuple> *index = new cCompleteRTree<cTuple>(indexName.c_str(), indexColumnPosition, blockSize, indexSD, indexSD->GetTypeSize(), sizeOfData, false, dsMode, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);


	if (index != NULL)
	{

		if ((implicitKeyVarlen == false && varlenData == false) || homogenous)//ověření zda se má používat fix pole dat nebo var pole dat
		{


			seqHeapFix->mSeqArray->OpenContext(seqHeapFix->mHeader->GetFirstNodeIndex(), 0, seqHeapFix->context);

			do
			{

				cTuple *heapTuple = new cTuple(SD);
				heapTuple->SetData(seqHeapFix->context->GetItem());
				cTuple *tuple = NULL;


				tuple = TransportItemFixLen(heapTuple, indexSD, indexColumnPosition, indexType);//problem s duplicitními záznamy(tomu se asi nevyhneme) jinak funkční


				int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapFix->context->GetNode()->GetIndex(), seqHeapFix->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapFix->mSeqArray->Advance(seqHeapFix->context));

			seqHeapFix->mSeqArray->CloseContext(seqHeapFix->context);

			return true;



		}
		else
		{
			seqHeapVar->mSeqArray->OpenContext(seqHeapVar->mHeader->GetFirstNodeIndex(), 0, seqHeapVar->context);

			do
			{
				//(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
				cHNTuple *heapTuple = new cHNTuple();
				heapTuple->Resize(SD);
				cTuple *tupleFix = NULL;
				cHNTuple *tupleVar = NULL;

				heapTuple->SetData(seqHeapVar->context->GetItem());

				if (varlenKey == false)
				{
					tupleFix = TransportItemFixLen(heapTuple, indexSD, indexColumnPosition, indexType);//problem s duplicitními záznamy(tomu se asi nevyhneme) jinak funkční
				}


				//int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tupleFix, LoadIndexData(seqHeapVar->context->GetNode()->GetIndex(), seqHeapVar->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapVar->mSeqArray->Advance(seqHeapVar->context));

			seqHeapVar->mSeqArray->CloseContext(seqHeapVar->context);

			return true;
		}
	}
	else
	{
		printf("Index: creation failed!\n");
		return false;
	}


}
inline bool cTable::ConstructIndexRtree(string indexName, cDataType * indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, cSpaceDescriptor *indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, cQuickDB * quickDB)
{
	unsigned int sizeOfData = sizeof(uint) + sizeof(short);
	cCompleteRTree<cTuple> *index = new cCompleteRTree<cTuple>(indexName.c_str(), indexColumnPosition, blockSize, indexSD, indexSD->GetTypeSize(), sizeOfData, false, dsMode, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);


	if (index != NULL)
	{

		if ((implicitKeyVarlen == false && varlenData == false) || homogenous)//ověření zda se má používat fix pole dat nebo var pole dat
		{


			seqHeapFix->mSeqArray->OpenContext(seqHeapFix->mHeader->GetFirstNodeIndex(), 0, seqHeapFix->context);

			do
			{

				cTuple *heapTuple = new cTuple(SD);
				heapTuple->SetData(seqHeapFix->context->GetItem());
				cTuple *tuple = NULL;

				
					tuple = TransportItemFixLen(heapTuple, indexSD, indexColumnPosition, indexType);//problem s duplicitními záznamy(tomu se asi nevyhneme) jinak funkční
				

				int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapFix->context->GetNode()->GetIndex(), seqHeapFix->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapFix->mSeqArray->Advance(seqHeapFix->context));

			seqHeapFix->mSeqArray->CloseContext(seqHeapFix->context);
			indexesFixLenRTree->push_back(index);

			return true;



		}
		else
		{
			seqHeapVar->mSeqArray->OpenContext(seqHeapVar->mHeader->GetFirstNodeIndex(), 0, seqHeapVar->context);

			do
			{
				//(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
				cHNTuple *heapTuple = new cHNTuple();
				heapTuple->Resize(SD);
				cTuple *tupleFix = NULL;
				cHNTuple *tupleVar = NULL;

				heapTuple->SetData(seqHeapVar->context->GetItem());

				if (varlenKey == false)
				{
					tupleFix = TransportItemFixLen(heapTuple, indexSD, indexColumnPosition, indexType);//problem s duplicitními záznamy(tomu se asi nevyhneme) jinak funkční
				}
				

				//int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tupleFix, LoadIndexData(seqHeapVar->context->GetNode()->GetIndex(), seqHeapVar->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapVar->mSeqArray->Advance(seqHeapVar->context));

			seqHeapVar->mSeqArray->CloseContext(seqHeapVar->context);

			return true;
		}
	}
	else
	{
		printf("Index: creation failed!\n");
		return false;
	}
}

inline bool cTable::ConstructIndexBtreeVar(string indexName, cDataType * indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, cSpaceDescriptor * indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, const uint inMemCacheSize, cQuickDB * quickDB)
{
	unsigned int sizeOfData = sizeof(uint) + sizeof(short);
	cCompleteBTree<cHNTuple> *index = new cCompleteBTree<cHNTuple>(indexName.c_str(), indexColumnPosition, blockSize, indexSD, indexSD->GetTypeSize(), sizeOfData, false, dsMode, cDStructConst::BTREE, compressionRatio, codeType, runtimeMode, histograms,  inMemCacheSize, quickDB);


	if (index != NULL)
	{

		if ((implicitKeyVarlen == false && varlenData == false) || homogenous)
		{


			seqHeapFix->mSeqArray->OpenContext(seqHeapFix->mHeader->GetFirstNodeIndex(), 0, seqHeapFix->context);

			do
			{

				cTuple *heapTuple = new cTuple(SD);
				heapTuple->SetData(seqHeapFix->context->GetItem());
				cHNTuple *tuple = NULL;

				if (homogenous && varlenKey)
				{
					tuple = TransportItemHomoVarlenCHNTuple(heapTuple, indexKeyColumnSD, indexSD, indexColumnPosition, indexType);
				}


				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapFix->context->GetNode()->GetIndex(), seqHeapFix->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapFix->mSeqArray->Advance(seqHeapFix->context));

			seqHeapFix->mSeqArray->CloseContext(seqHeapFix->context);

			return true;



		}
		else
		{
			seqHeapVar->mSeqArray->OpenContext(seqHeapVar->mHeader->GetFirstNodeIndex(), 0, seqHeapVar->context);

			do
			{
				//(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
				cHNTuple *heapTuple = new cHNTuple();
				heapTuple->Resize(SD);

				cHNTuple *tuple = NULL;

				heapTuple->SetData(seqHeapVar->context->GetItem());

				if (varlenKey)
				{

					tuple = TransportItemVarLenCHNTuple(heapTuple, indexKeyColumnSD, indexSD, indexColumnPosition, indexType);
				}

				//int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapVar->context->GetNode()->GetIndex(), seqHeapVar->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapVar->mSeqArray->Advance(seqHeapVar->context));

			seqHeapVar->mSeqArray->CloseContext(seqHeapVar->context);

			return true;
		}
	}
	else
	{
		printf("Index: creation failed!\n");
		return false;
	}
}

inline bool cTable::ConstructIndexRtreeVar(string indexName, cDataType * indexType, int indexColumnPosition, uint blockSize, cSpaceDescriptor * indexSD, cSpaceDescriptor * indexKeyColumnSD, bool varlenKey, uint dsMode, uint compressionRatio, unsigned int codeType, unsigned int runtimeMode, bool histograms, cQuickDB * quickDB)
{
	unsigned int sizeOfData = sizeof(uint) + sizeof(short);
	cCompleteRTree<cHNTuple> *index = new cCompleteRTree<cHNTuple>(indexName.c_str(), indexColumnPosition, blockSize, indexSD, indexSD->GetTypeSize(), sizeOfData, false, dsMode, cDStructConst::RTREE, compressionRatio, codeType, runtimeMode, histograms, quickDB);


	if (index != NULL)
	{

		if ((implicitKeyVarlen == false && varlenData == false) || homogenous)
		{


			seqHeapFix->mSeqArray->OpenContext(seqHeapFix->mHeader->GetFirstNodeIndex(), 0, seqHeapFix->context);

			do
			{

				cTuple *heapTuple = new cTuple(SD);
				heapTuple->SetData(seqHeapFix->context->GetItem());
				cHNTuple *tuple = NULL;

				if (homogenous && varlenKey)
				{
					tuple = TransportItemHomoVarlenCHNTuple(heapTuple, indexKeyColumnSD, indexSD, indexColumnPosition, indexType);
				}


				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapFix->context->GetNode()->GetIndex(), seqHeapFix->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapFix->mSeqArray->Advance(seqHeapFix->context));

			seqHeapFix->mSeqArray->CloseContext(seqHeapFix->context);

			return true;



		}
		else
		{
			seqHeapVar->mSeqArray->OpenContext(seqHeapVar->mHeader->GetFirstNodeIndex(), 0, seqHeapVar->context);

			do
			{
				//(cHNTuple * sourceTuple, cSpaceDescriptor * columnSD, cSpaceDescriptor * keySD, int columnPosition, cDataType * mType)
				cHNTuple *heapTuple = new cHNTuple();
				heapTuple->Resize(SD);

				cHNTuple *tuple = NULL;

				heapTuple->SetData(seqHeapVar->context->GetItem());

				if (varlenKey)
				{
								
					tuple = TransportItemVarLenCHNTuple(heapTuple, indexKeyColumnSD, indexSD, indexColumnPosition, indexType);
				}

				//int a = tuple->GetInt(0, indexSD);

				bool hm = index->mIndex->Insert(*tuple, LoadIndexData(seqHeapVar->context->GetNode()->GetIndex(), seqHeapVar->context->GetPosition()));
				if (hm)
				{
					cout << "value inserted" << endl;
				}
				else
					cout << "value not inserted" << endl;
			} while (seqHeapVar->mSeqArray->Advance(seqHeapVar->context));

			seqHeapVar->mSeqArray->CloseContext(seqHeapVar->context);

			return true;
		}
	}
	else
	{
		printf("Index: creation failed!\n");
		return false;
	}
}

inline cTreeItemStream<cTuple>* cTable::Find(int key)
{
	cSpaceDescriptor *findSD = new cSpaceDescriptor(1, new cTuple(), new cInt());
	cTuple *findTuple = new cTuple(findSD);
	findTuple->SetValue(0, key, findSD);

	cRangeQueryConfig *config = new cRangeQueryConfig();

	//config->SetFinalResultSize(1);
	//config->SetQueryProcessingType(QueryType::SINGLEQUERY);
	config->SetBulkReadEnabled(false);

	config->SetFinalResultSize(0);

	
	cTreeItemStream<cTuple>*result =indexesFixLenRTree->at(1)->mIndex->RangeQuery(*findTuple, *findTuple, config);

	return result;

}

inline bool cTable::Serialization()
{

	/*cInt ii;
	for (int i = 0; i < columns->size; i++)
	{
		cColumn *column = columns->at(i);
		switch (column->cType->GetCode())
		{
		case 'i':
			ii = *(cInt*)column->cType;
			break;
		default:
			break;
		}
		
		

	}*/



	/*serializace*/
	std::ofstream ofs("SystemCatalog.dat");
	

	sSystemCatalog systemCatalog;

	Base* baseClass = new Child2();
	baseClass->baseInt = 269;

	systemCatalog.baseClass = baseClass;




	systemCatalog.tableName = tableName;
	systemCatalog.typeOfTable = typeOfTable;
	////systemCatalog.smart_columns = shared_ptr <vector<cColumn*>>(columns);
	systemCatalog.varlenData = varlenData;
	systemCatalog.implicitKeyVarlen = implicitKeyVarlen;
	systemCatalog.homogenous = homogenous;
	systemCatalog.keySD = keySD;
	systemCatalog.keyType = keyType;
	systemCatalog.varlenKeyColumnSD = varlenKeyColumnSD;

	boost::archive::text_oarchive oa(ofs);
	oa.register_type<cInt>();
	//oa.register_type<cBasicType<int>>();
	// write class instance to archive
	oa << systemCatalog;

	
	return true;
}


inline char * cTable::LoadIndexData(unsigned int nodeId, unsigned int position)
{
	ushort x = (short)position;
	uint y = nodeId;

	char * data = static_cast<char*>(static_cast<void*>(&x));
	char * data2 = static_cast<char*>(static_cast<void*>(&y));
	char *datas[] = { data, data2 };
	char sizes[] = { sizeof x, sizeof y };
	char bytes[6];
	cDataType *types[] = { new cUShort(),new cUInt() };



	for (size_t i = 0; i < 2; i++)
	{
		int offset = 0;
		for (size_t k = 0; k < i; k++)
		{
			offset += types[k]->GetSize();
		}
		for (size_t j = 0; j < types[i]->GetSize(); j++)
		{
			char * tmp = datas[i];

			bytes[i *  offset + j] = tmp[j];
		}
	}

	return bytes;
}




