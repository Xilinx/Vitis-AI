#!/bin/bash

# Check if all source files have the correct license

LICENSE=$1
TYPES="c cpp h cl"
IGNORE=$(cat .LICENSE_IGNORE.txt)

LICENSE_LEN=$(cat LICENSE.txt | wc -l)

echo "-------------------------------------"
echo "--  CHECKING LICENSE of all $TYPES --"
echo "-------------------------------------"
echo "-- IGNORING "
echo "$IGNORE"
echo "-----------"

FAIL=0

check_file() {
	ignore=0

	for i in $IGNORE; do
		if [[ $1 =~ $i ]]; then 
			ignore=1
		fi
	done

	if [[ $VERBOSE == "true" ]]; then
		echo -n "Checking $1 ... "
	fi
	if [[ $ignore == 1 ]]; then
		if [[ $VERBOSE == "true" ]]; then
			echo "SKIP"
		fi
	else
		diff $LICENSE <(head -n$LICENSE_LEN $1) 2>/dev/null 1>&2
		if [[ $? == 0 ]]; then
			if [[ $VERBOSE == "true" ]]; then
				echo "PASS"
			fi
		else
			if [[ $VERBOSE == "true" ]]; then
				echo "FAIL"
				diff $LICENSE <(head -n$LICENSE_LEN $1)
			else
				echo "$1"
			fi
			(( FAIL += 1 ))
		fi
	fi
}


VCS_FILES=$(git ls-files)

for f in $VCS_FILES; do
	for t in $TYPES; do
		if [[ $f == *.$t ]]; then
			check_file $f
		fi
	done
done

if [[ $FAIL != 0 ]]; then
	echo "ERROR: License check failed"
	echo "ERROR: please fix the license in these files (or add to ignored if external)"
fi

exit $FAIL
