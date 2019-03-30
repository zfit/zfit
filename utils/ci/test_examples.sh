for file in examples/*; do
    python $file 2>&1 | tail -n 11 && echo "file $file run sucessfully";
done
