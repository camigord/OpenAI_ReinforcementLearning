import numpy as np
import math

class BinaryHeap(object):
    def __init__(self,size):
        self.e2p = {}
        self.p2e = {}
        self.priority_queue = {}
        self.size = 0
        self.max_size = size
        
    def __repr__(self):
        '''
        Just for testing purposes
        Returns string of the priority queue
        '''
        if self.size == 0:
            return 'No element in heap!'
        
        to_string = ''
        level = -1
        max_level = math.floor(math.log(self.size,2))
        
        for i in range(1,self.size+1):
            current_level = math.floor(math.log(i,2))
            if level != current_level:
                to_string = to_string + ('\n' if level != -1 else '') + '    ' * int(max_level-current_level)
                level = current_level
                
            to_string = to_string + '%.2f ' % self.priority_queue[i][1] + '    '*int(max_level-current_level)
            
        return to_string
    
    def _insert(self,priority,e_id):
        '''
        Inserts new experience id with priority
        '''
        self.size = min(self.size+1,self.max_size)       # Replacing exp. with lowest priority
        self.priority_queue[self.size] = (priority,e_id) 
        self.p2e[self.size] = e_id                       # Auxiliar index to determine location of exp. IDs
        self.e2p[e_id] = self.size
        
        self.up_heap(self.size)                          # Percolating up
        
    def update(self,priority,e_id):
        '''
        Updates priority value of a given experience ID
        '''
        if e_id in self.e2p:
            p_id = self.e2p[e_id]                        # Get position of exp. ID
            self.priority_queue[p_id] = (priority,e_id)  # Replace priority of that experience
            self.p2e[p_id] = e_id                        # Store exp id stores in p_id
            self.down_heap(p_id)                         # Percolating down
            self.up_heap(p_id)                           # Percolating up
        else:
            self._insert(priority,e_id)       # Insert new ID     
            
    def get_max_priority(self):
        '''
        Gets max. priority. If empty, returns 1
        '''
        return self.priority_queue[1][0] if self.size > 0 else 1
    
    def up_heap(self,i):
        '''
        Percolation up
        '''
        if i>1:
            parent = math.floor(i/2)
            if self.priority_queue[parent][0] < self.priority_queue[i][0]: # If child is larger than parent
                # Swapping positions
                self._swap_positions(parent,i)
                # Percolating up the parent (which once we get here is equal to our previous child)
                self.up_heap(parent)
                
    def down_heap(self,i):
        '''
        Percolation down
        '''
        if i<self.size:
            greatest = i
            left,right = i*2, i*2+1               # Index of both children
            
            # Check if a child node is larger. Determine which one is largest
            if left < self.size and self.priority_queue[left][0] > self.priority_queue[greatest][0]:
                greatest = left
            if right < self.size and self.priority_queue[right][0] > self.priority_queue[greatest][0]:
                greatest = left
                
            if greatest != i:                     # If a children is larger
                self._swap_positions(i,greatest)
                # Percolating down the new child (previous parent)
                elf.down_heap(greatest)
            
    def _swap_positions(self,parent,child):
        tmp = self.priority_queue[child]
        self.priority_queue[child] = self.priority_queue[parent]
        self.priority_queue[parent] = tmp
        # Change auxiliars e2p, p2e
        self.e2p[self.priority_queue[child][1]] = child
        self.e2p[self.priority_queue[parent][1]] = parent
        self.p2e[child] = self.priority_queue[child][1]
        self.p2e[parent] = self.priority_queue[parent][1]
        
    def balance_tree(self):
        '''
        Rebalances priority queue
        '''
        sort_array = sorted(self.priority_queue.values(), key=lambda x: x[0], reverse=True)
        # Reconstruct priority queue
        self.priority_queue.clear()
        self.p2e.clear()
        self.e2p.clear()
        count = 1
        while count <= self.size:
            priority, e_id = sort_array[count-1]
            self.priority_queue[count] = (priority,e_id)
            self.p2e[count] = e_id
            self.e2p[e_id] = count
            count += 1
        
        # Sort the heap
        for i in range(int(math.floor(self.size/2)),1,-1):
            self.down_heap(i)