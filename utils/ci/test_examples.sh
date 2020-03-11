pip install matplotlib
set -e
for file in examples/*; do
    python $file
#    below needed?
#    python $file 2>&1 | tail -n 11 && echo "file $file run sucessfully" || exit 1;
done
