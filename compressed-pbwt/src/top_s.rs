pub fn select_top_s<T>(s: usize, rank: Vec<T>) -> Vec<T>
where
    T: std::cmp::Ord + std::cmp::Eq + Clone,
{
    let mut top_s = std::collections::BinaryHeap::<T>::new();
    for r in rank {
        if top_s.len() == s {
            if r < *top_s.peek().unwrap() {
                top_s.pop().unwrap();
                top_s.push(r);
            }
        } else {
            top_s.push(r);
        }
    }
    let mut top_s = top_s.into_iter().collect::<Vec<_>>();
    top_s.sort();
    top_s
}

// assume the ranks are sorted
pub fn merge_top_s<T: std::cmp::PartialOrd>(s: usize, rank_a: &[T], rank_b: &[T]) -> Vec<T>
where
    T: std::cmp::PartialOrd + Clone,
{
    let mut top_s = Vec::with_capacity(s);
    let mut rank_a_iter = rank_a.iter();
    let mut rank_b_iter = rank_b.iter();
    let mut a = rank_a_iter.next();
    let mut b = rank_b_iter.next();
    for _ in 0..s {
        match (a, b) {
            (Some(&ref a_), Some(&ref b_)) => {
                if a_ < b_ {
                    top_s.push(a_.clone());
                    a = rank_a_iter.next();
                } else {
                    top_s.push(b_.clone());
                    b = rank_b_iter.next();
                }
            }
            (Some(&ref a_), None) => {
                top_s.push(a_.clone());
                a = rank_a_iter.next();
            }
            (None, Some(&ref b_)) => {
                top_s.push(b_.clone());
                b = rank_b_iter.next();
            }
            (None, None) => break,
        }
    }
    top_s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_merge_top_s() {
        let s = 4;
        let rank = &[(3, 0), (1, 0), (5, 0), (2, 0), (4, 0)];
        assert_eq!(select_top_s(s, rank), &[(1, 0), (2, 0), (3, 0), (4, 0)]);
        let rank_a = &[(1, 0), (3, 0), (5, 0), (7, 0)];
        let rank_b = &[(2, 0), (4, 0), (6, 0), (8, 0)];
        assert_eq!(
            merge_top_s(s, rank_a, rank_b),
            &[(1, 0), (2, 0), (3, 0), (4, 0)]
        );
    }
}
