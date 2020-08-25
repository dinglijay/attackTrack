#! /bin/sh
for file in ./lasotconfig/*.ini
do
   if [[ -f "$file"  ]]; then
      echo $file
      python experiment/ater.py $file # process the file with name in "$file" here
   fi
done