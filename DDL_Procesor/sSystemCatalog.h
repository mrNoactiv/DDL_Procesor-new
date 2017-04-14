#include "cTranslatorCreate.h"
#include "boost\serialization\export.hpp"



class Base {

public:
	int baseInt;


};
template <class T>
class Child1 :public Base {
public:
	int Child1Int;

};

class Child2 : public Child1<Child2> {

};


class sSystemCatalog {

public:
	Base* baseClass;
	
	string tableName;
	TypeOfCreate typeOfTable;
	vector<cColumn*>*columns;
	bool varlenData;
	bool implicitKeyVarlen;
	bool homogenous;
	cSpaceDescriptor *keySD;
	cDataType* keyType;
	cSpaceDescriptor *varlenKeyColumnSD;
	
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & baseClass;
		//
		//
		//ar & tableName;
		//ar & typeOfTable;
		//ar & varlenData;
		//ar & implicitKeyVarlen;
		//ar & homogenous;
		//ar & keySD;
		ar & *keyType;

		ar & homogenous;

	}



};


namespace boost {
	namespace serialization {

		template<class Archive>
		inline void serialize(
			Archive & ar,
			Base & t,
			const unsigned int file_version
		) {
			ar & t.baseInt;
		}
		
		
		template<class Archive>
		inline void serialize(
			Archive & ar,
			cSpaceDescriptor & t,
			const unsigned int file_version
		) {
			ar & t.GetDimension();
		}

		template<class Archive>
		inline void serialize(
			Archive & ar,
			cDataType & t,
			const unsigned int file_version
		) {
			printf("Called CDataType\n");
			ar &  112;
		}
		//template<class Archive, class T>
		//inline void serialize(
		//	Archive & ar,
		//	cBasicType<T> & t,
		//	const unsigned int file_version
		//) {
		//	ar & boost::serialization::base_object<cDataType>(*t);;
		//}

		template<class Archive>
		inline void serialize(
			Archive & ar,
			cInt & t,
			const unsigned int file_version
		) {
			printf("Called cInt\n");
			ar & "potato";
		}
	}
}

//BOOST_CLASS_EXPORT(cInt)




