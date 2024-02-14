## declare an array variable
declare -a arr=("m3vcf"
"common"
"compressed-pbwt"
"host"
"obliv-utils"
"phasing"
"tp-fixedpoint"
"switch-error-check"
)

for dir in "${arr[@]}"
do
    (cd $dir && cargo clean)
done


