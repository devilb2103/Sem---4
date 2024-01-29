package Java.Assignments.Assignment2;

import java.lang.reflect.Array;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;

public class arrayClass {
    public int even[];
    public int odd[];
    public int nums[];
    private int pos_even = 0, pos_odd = 0, pos_nums = 0;

    // getter setter ---------------------
    
    public String getEven(){
        return Arrays.toString(this.even);
    }

    public String getOdd(){
        return Arrays.toString(this.odd);
    }
    
    // utilit functions ---------------------
    arrayClass(int size){
        this.even = new int[size];
        this.odd = new int[size];
        this.nums = new int[size];
    }

    public void appendEven(int x){
        this.even[this.pos_even] = x;
        this.pos_even += 1;
    }

    public void appendOdd(int x){
        this.odd[this.pos_odd] = x;
        this.pos_odd += 1;
    }

    public void appendNums(int x){
        this.nums[this.pos_nums] = x;
        this.pos_nums += 1;
    }

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

    public ArrayList<Integer> arrayToArrayList(int arr[]){
        ArrayList<Integer> arrList = new ArrayList();

        for (int i = 0; i < arr.length; i++) {
            arrList.add(arr[i]);
        }

        return arrList;
    }

    public int[] arrayToArrayList(ArrayList<Integer> arr){
        return arr.toArray();
    }

}
