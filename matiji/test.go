package main

import (
	"fmt"
)
func main() {
	var n,a,b,len int 
	fmt.Scan(&n)
	numlist:=make([]int,2000000)
	for i:=0;i<n;i++{
		fmt.Scan(&a)
		if a==1||a==3{
			fmt.Scan(&b)
		}
		switch a{
		case 1:numlist[len]=b;len++
		case 2:fmt.Println(numlist[len-1])
		case 3:fmt.Println(numlist[b])
		case 4:len--
		}
	}
}
