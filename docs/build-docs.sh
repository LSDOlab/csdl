#!/bin/bash

# install package and dependencies
apt-get -y install python3-pip nodejs npm
pip3 install redbaron
# install yapf to format examples for docs
pip3 install yapf
# install backend to run examples in docs
git clone --depth=1 https://github.com/lsdolab/csdl_om.git
pip3 install -e csdl_om/
pip3 install -e .

# build docs
npm run build
here=`pwd`
mkdir -p ${here}/docs/_build/html/examples
python3 ${here}/docs/utils/clean_examples.py ${here}
ls ${here}/docs/_build/html/examples
make -C docs html
ls ${here}/docs/_build/html/

echo "DOCS BUILT"

docroot=`mktemp -d`
# copy docs to temporary directory
rsync -av "${here}/docs/build/ "${docroot}"

# create git repository in temporary directory
pushd "${docroot}"
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages

# add .nojekyll so GitHub generates site from docs
touch .nojekyll

# add README for gh-pages branch
cat > README.md <<EOF
# csdl
This branch contains the generated documentation pages for
[csdl](https://lsdolab.github.io/csdl).
EOF

# commit changes
git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
ls -a
git add .
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
msg="Updating Docs for commit ${GITHUB_SHA} from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"

# update gh-pages
git push deploy gh-pages --force
