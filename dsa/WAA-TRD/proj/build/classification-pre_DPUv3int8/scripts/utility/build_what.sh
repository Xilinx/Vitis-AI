#!/bin/bash -e

HEAD=

if [[ "$CHANGE_TARGET" == "" ]]; then
	HEAD=remotes/origin/master
else
	HEAD=remotes/origin/${CHANGE_TARGET}
fi

PROJS=$(git ls-files | grep description.json | sed -e 's/\.\///' -e 's/\/description.json//')
CHANGES=$(git diff --name-only $HEAD)

howmany() { echo $#; }
NUM_CHANGES=$(howmany $CHANGES)

echo NUM_CHANGES=$NUM_CHANGES

REBUILDS=
for change in $CHANGES; do
	IN_PROJS=
	for proj in $PROJS; do
		if [[ "$change" == ${proj}* ]]; then
			IN_PROJS="$proj $IN_PROJS"
		fi
	done

	if [[ "$change" == */README.md
		|| "$change" == "utility/build_what.sh"
		|| "$change" == "Jenkinsfile" ]]; then
		echo "SKIPPING $change"
		NUM_CHANGES=$((NUM_CHANGES-1))
	elif [[ "$IN_PROJS" != "" ]]; then
		echo "REBUILD $change"
		NUM_CHANGES=$((NUM_CHANGES-1))
		REBUILDS="$IN_PROJS $REBUILDS"
	else
		echo "UNKNOWN $change"
	fi
done

UNIQ_REBUILDS=$(echo $REBUILDS | xargs -n 1 | sort -u | xargs)

echo UNIQ_REBUILDS = $UNIQ_REBUILDS
echo NUM_CHANGES = $NUM_CHANGES

# if we know that we only changed something inside a single example then do a rebuild
# of that example only else rebuild all examples.
cat /dev/null > examples.dat
if [[ "$NUM_CHANGES" == "0" && "$UNIQ_REBUILDS" != "" ]]; then
	for rebuild in $UNIQ_REBUILDS; do
		echo $rebuild >> examples.dat
	done
else
	for proj in $PROJS; do
		echo $proj >> examples.dat
	done
fi
