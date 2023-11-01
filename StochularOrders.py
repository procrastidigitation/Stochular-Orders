# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:58:35 2023

@author: Vishal
"""

from itertools import accumulate as acc
from numpy import isclose
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


class Distribution:
    
    def __init__(self, vals, probs, precision=16):
        """docstring"""
        
        #Decimal precision for probs
        self.precision = precision
        
        #Anti-cloning the vals
        merge_vec = {v:0.0 for v in vals}
        for i in range(len(vals)):
            merge_vec[vals[i]] += probs[i]
        merge_vec = sorted(list(merge_vec.items()))
        
        #Dist data
        self.vals = [v for (v,p) in merge_vec]
        self.rawprobs = [p for (v,p) in merge_vec]
        self.probs = [p for (v,p) in merge_vec]
        self.renorm()
        self.dist = list(zip(self.vals, self.probs))       
        
        #CDF data
        self.cumu_probs = list(acc(self.probs))
        self.cumu_probs[-1] = 1
        
        #Quantile data
        self.quants = self.cumu_probs
        self.quant_vals = self.vals
        
        #Discrete quantile integrals. Memoising integral.
        self.quints = {(l,r):sum([(self.quants[i] - self.quants[i-1])*self.quant_vals[i] for i in range(l, r)]) for l in range(1,len(self.quants)) for r in range(l,len(self.quants))}
        
        #LUB Map. Memoising integral.
        self.lubs = {}

        
    def renorm(self):
        """docstring"""
        
        #Enforcing decimal precision
        self.probs = [round(p, self.precision) for p in self.probs]
        
        #Ensuring sum to 1
        if sum(self.probs) != 1.0:
            self.probs[-1] = 1.0 - sum(self.probs[:-1])
            
        assert all(p > 0.0 for p in self.probs), "Some probabilities are not positive"
        assert sum(self.probs) == 1.0, "Probabilities do not sum to 1"

        
    def cdf(self, v):
        """docstring"""
        
        for i in range(len(self.vals)-1, -1, -1):
            if self.vals[i] <= v:
                return self.cumu_probs[i]
        return 0.0

    
    def quantile(self, q):
        """ docstring"""
        
        for i in range(len(self.quants)):
            if self.quants[i] >= q:
                return self.quant_vals[i]
            
    
    def expectation(self):
        """docstring"""
        
        weighted_vals = [self.vals[i]*self.probs[i] for i in range(len(self.vals))]
        return round(sum(weighted_vals), int(self.precision**0.5))
    
    
    def median(self):
        """docstring"""
        
        return self.quantile(0.5)
    
    
    def moment(self, n):
        """docstring"""
        
        weighted_vals = [(self.vals[i]**n)*self.probs[i] for i in range(len(self.vals))]
        return round(sum(weighted_vals), int(self.precision**0.5))
    
    
    def central_moment(self, n):
        """docstring"""
        
        E = self.expectation()
        weighted_vals = [((self.vals[i]-E)**n)*self.probs[i] for i in range(len(self.vals))]
        return round(sum(weighted_vals), int(self.precision**0.5))
            

    def translate(F, t):
        """docstring"""
        
        vals = [v+t for v in F.vals]
        return Distribution(vals, F.probs, F.precision)
    
    
    def scale(F, c):
        """docstring"""
        
        vals = [c*v for v in F.vals]
        return Distribution(vals, F.probs, F.precision)
    
    
    def indep_add(F, G):
        """docstring"""
        
        vals = [v1+v2 for v1 in F.vals for v2 in G.vals]
        probs = [p1*p2 for p1 in F.probs for p2 in G.probs]
        return Distribution(vals, probs, max(F.precision, G.precision))
    
    
    def comon_add(F, G):
        """docstring"""
        
        quants = sorted(list(set(F.quants + G.quants)))
        quant_vals = [F.quantile(q) + G.quantile(q) for q in quants]
        return Distribution(quant_vals, [quants[0]] + [quants[i]-quants[i-1] for i in range(1, len(quants))], max(F.precision, G.precision))
    
    
    def mixture(F, G, alpha):
        """docstring"""
        
        vals = F.vals + G.vals
        probs = [alpha*p for p in F.probs] + [(1.0-alpha)*p for p in G.probs]
        return Distribution(vals, probs, max(F.precision, G.precision))
    
    
    def get_q_lub_ind(self, q):
        """docstring"""
        
        if q not in self.lubs:
            i = 0
            while self.quants[i] < q:
                i += 1
            self.lubs[q] = i
        return self.lubs[q]
    
    
    def integral(self, a, b):
        """docstring"""
        
        #Returns value of the quantile function if a==b
        if a == b:
            return self.quantile(a)
        
        #Else returns the integral of the quantile function from a to b
        #l = r = 0
        #while self.quants[l] < a:
        #    l += 1
        #while self.quants[r] < b:
        #    r += 1
        l = self.get_q_lub_ind(a)
        r = self.get_q_lub_ind(b)
            
        if l == r:
            return round((b-a)*self.quant_vals[l], int(self.precision**0.5))
        
        l_margin = (self.quants[l] - a)*self.quant_vals[l]
        r_margin = (b - self.quants[r-1])*self.quant_vals[r]
        #return l_margin + sum([(self.quants[i] - self.quants[i-1])*self.quant_vals[i] for i in range(l+1, r)]) + r_margin
        return round(l_margin + self.quints[(l+1,r)] + r_margin, int(self.precision**0.5))
    
    
    #Presently incomplete
    def quantile_render(F, G):
        """docstring"""
        ax = plt.gca()
        
        plt.plot([0,1], [0,0], alpha=0.5, color='black', linewidth=0.5, linestyle=':')
        
        plt.step([0.0]+F.quants, [F.quant_vals[0]]+F.quant_vals, label='F', alpha=0.9)
        ax.fill_between([0.0]+F.quants, 0, [F.quant_vals[0]]+F.quant_vals, alpha=0.2, step='pre')
        
        plt.step([0.0]+G.quants, [G.quant_vals[0]]+G.quant_vals, label='G', alpha=0.9)
        ax.fill_between([0.0]+G.quants, 0, [G.quant_vals[0]]+G.quant_vals, alpha=0.2, step='pre')
        
        ax.legend()
        ax.set_xlabel('q')
        ax.set_ylabel('v', rotation='horizontal')
        #ax.set_aspect('equal')
        #ax.spines['bottom'].set_position('zero')
        #ax.spines['left'].set_position('zero')
        #ax.spines['top'].set_visible(False)
        #ax.spines['right'].set_visible(False)
        plt.show()
        
        
class Bridge:
    
    def __init__(self, phi, fineness=0.005):
        """docstring"""
        
        #Grid fineness
        self.fineness = fineness
        
        #Phi is the generating function
        self.phi = phi
        
    
    def adjust_fineness(self, fineness):
        """docstring"""
        
        self.fineness = fineness
        
    
    def reach(self, F, G):
        """docstring"""
        
        F = Distribution.scale(F, 1/self.fineness)
        G = Distribution.scale(G, 1/self.fineness)
        
        #To stay within the triangle
        def clamp(x):
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x
        
        #Finding minimum value of phi under which F is smaller than G
        min_fail = float("inf")
        for b_ind in range(int(1.0/self.fineness)):
            for a_ind in range(b_ind+1):
                b = clamp(b_ind * self.fineness)
                a = clamp(a_ind * self.fineness)
                #if F.integral(a, b) < G.integral(a, b) and not isclose(F.integral(a, b), G.integral(a, b)):
                if F.integral(a, b) < G.integral(a, b):
                    min_fail = min(min_fail, self.phi(a,b))
        
        #If F was never smaller than G, return reach of 1.0           
        if min_fail == float("inf"):
            return 1.0
        
        #Finding maximum value strictly below min_fail where F bigger than G has been achieved. This is the reach of F over G along self.
        max_succ = float("-inf")
        for b_ind in range(int(1.0/self.fineness)):
            for a_ind in range(b_ind+1):
                b = clamp(b_ind * self.fineness)
                a = clamp(a_ind * self.fineness)
                #if (F.integral(a, b) >= G.integral(a, b) or isclose(F.integral(a, b), G.integral(a, b))) and self.phi(a,b) < min_fail:
                if F.integral(a, b) >= G.integral(a, b) and self.phi(a,b) < min_fail:
                    max_succ = max(max_succ, self.phi(a,b))
        
        return max_succ


    def heatmap_render(F, G, fineness=0.005):
        """docstring"""
        
        #Don't forget to zero out the diagonal
        d = int(1.0/fineness)
        M = [[0.0 for _ in range(d)] for _ in range(d)]
        
        #To stay within the triangle
        def clamp(x):
            if x < 0.0:
                return 0.0
            if x > 1.0:
                return 1.0
            return x
        
        max_int = float("-inf")
        min_int = float("inf")
        for b_ind in range(d):
            for a_ind in range(b_ind):
                b = clamp(b_ind * fineness)
                a = clamp(a_ind * fineness)
                M[b_ind][a_ind] = F.integral(a,b) - G.integral(a,b)
                max_int = max(max_int, M[b_ind][a_ind])
                min_int = min(min_int, M[b_ind][a_ind])
                
        norm = mcolors.TwoSlopeNorm(0)
        cmap = cm.get_cmap('coolwarm_r')
        if max_int == 0:
            cmap = cm.get_cmap('Reds_r')
            norm = None
        if min_int == 0:
            cmap = cm.get_cmap('Blues')
            norm = None
        plt.imshow(M, interpolation="nearest", cmap=cmap, norm=norm, origin='lower')
        plt.axis('off')
        #plt.colorbar()
        plt.show()