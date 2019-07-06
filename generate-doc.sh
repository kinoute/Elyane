#/bin/sh

echo "Temporary renaming the python files in the root directory..."

for script in *.py; do
    mv $script "$script.bak"
done

echo "Generating the docs..."

rm -rf docs
pdoc3 --html . --force -o docs

echo 'Removing pycache folders...'

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

echo 'Moving the documentation to the correct folder...'

mv docs/Elyane/* docs/
rm -rf docs/Elyane

echo "Renaming back the python files to the correct format..."

for script in *.bak; do
    mv $script "${script%.bak}"
done


