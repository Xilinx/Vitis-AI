#
# Copyright 2022-2023 Advanced Micro Devices Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

all_notebooks :=

all_notebooks += structure


publish_dir = /workspace/aisw/xmodel_image_notebooks


publish_files += $(publish_dir)/structure.md
publish_files += $(publish_dir)/xmodel_image.svg
publish_files += $(publish_dir)/vart_mat.svg
publish_files += $(publish_dir)/xmodel_image_imp.svg

all: debug

all: publish xmodel_image_imp.svg


.PHONY: publish

copy: $(publish_files)

publish: $(publish_files)
	@cd $(publish_dir) && git reset --soft tag_start && git add . && git commit -m 'update' && git push --force

$(publish_dir)/%.html : %.html
	@cp -av $< $@

$(publish_dir)/%.svg : %.svg
	@cp -av $< $@

$(publish_dir)/%.md : %.md
	@cp -av $< $@

$(publish_dir)/%.ipynb : %.ipynb
	@cp -av $< $@

%.html: %.ipynb
	jupyter nbconvert  \
    --template classic --to html $<

%.svg: %.uml
	plantuml -tsvg -v $<

%.ipynb: %.md
	jupytext  --execute --to notebook $<

clean:
	-rm $(foreach notebook,$(all_notebooks), $(publish_dir)/$(notebook).html) $(foreach notebook,$(all_notebooks), $(publish_dir)/$(notebook).ipynb)

debug:
	echo $(publish_files)
