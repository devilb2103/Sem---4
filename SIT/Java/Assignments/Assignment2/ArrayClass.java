package Java.Assignments.Assignment2;

import java.lang.Math;
import java.util.ArrayList;

public class ArrayClass {
    private int[] even, odd, nums;
    private int pos_even = 0, pos_odd = 0, pos_nums = 0;

    // getters and constructor ---------------------
    
    public int[] getEven(){
        return this.even;
    }

    public int[] getOdd(){
        return this.odd;
    }

    public int[] getArr(){
        return this.nums;
    }
    
    ArrayClass(int size){
        this.even = new int[size];
        this.odd = new int[size];
        this.nums = new int[size];
    }
    
    // utility functions ---------------------

    // Append to even array
    public void appendEven(int x){
        this.even[this.pos_even] = x;
        this.pos_even += 1;
    }

    // Append to odd array
    public void appendOdd(int x){
        this.odd[this.pos_odd] = x;
        this.pos_odd += 1;
    }

    // append to universal array
    public void appendNums(int x){
        this.nums[this.pos_nums] = x;
        this.pos_nums += 1;
    }

    // find smallest neighbouring distance implementation 
    public int[] findSmallestDistance(){
        int arr[] = nums.clone();
        
        int dist = Integer.MAX_VALUE;
        int pos = 0;
        for (int i = 0; i < arr.length - 1; i++) {
            int diff = Math.abs(arr[i] - arr[i + 1]);
            if(diff < dist){
                dist = diff;
                pos = i;
            }
        }

        int res[] = {dist, pos};

        return res;
    }

    // array to arrayList implementation
    public ArrayList<Integer> arrayToArrayList(int arr[]){
        ArrayList<Integer> arrList = new ArrayList<Integer>();

        for (int i = 0; i < arr.length; i++) {
            arrList.add(arr[i]);
        }

        return arrList;
    }

    // arrayList to array implementation
    public int[] ArrayListToArray(ArrayList<Integer> arrList){
        Object ObjArray[] = arrList.toArray();

        int arr[] = new int[ObjArray.length];

        for (int i = 0; i < ObjArray.length; i++) {
            arr[i] = (int)ObjArray[i];
        }

        return arr;
    }

}
