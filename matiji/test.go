package main

import (
	"fmt"
)
func main() {
	var n int
	var check bool
	fmt.Scan(&n)
	for i:=2;i<n;i++{
		check=true
		for ii:=0;ii<i;ii++{
			if i%ii==0{
				check=false
				}
			}
		if check{
			fmt.Println(i)
			}
		}
}
